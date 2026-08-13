"""ClaudeStreamJsonClient — wraps the `claude` CLI (Claude Code) as an
OpenAI-compatible chat.completions facade for Hermes delegation.

This is the transport for the 鲁班 (Luban = Claude Code CLI) line. Hermes
spawns `claude --print --input-format stream-json --output-format stream-json`
as a subprocess, feeds it a single user message, and reads the streamed JSON
events back. Unlike CopilotACPClient (which speaks ACP JSON-RPC), this client
speaks Claude Code's own stream-json event protocol:

  - input (one JSON line on stdin):
      {"type":"user","message":{"role":"user","content":[{"type":"text","text":"<goal>"}]}}
  - output (JSON lines on stdout):
      {"type":"system","subtype":"init",...}     # session metadata, ignored
      {"type":"assistant","message":{...}}        # streamed text fragments, ignored
      {"type":"result","result":"<final text>","subtype":"success",...}  # TERMINAL

The client waits for the `result` event and returns its `result` field as the
assistant content. Streaming is emulated (the subprocess runs to completion
before we emit OpenAI-style chunks) — same pattern CopilotACPClient uses, since
`claude --print` does not surface incremental OpenAI deltas.

Tools passed by run_agent are ACCEPTED BUT IGNORED: Claude Code has its own
native toolset (Bash/Read/Edit/...), and delegated goals are self-contained by
the delegate_task contract. We log this so it is auditable, never silent.

 ponytail: external subprocess transport — the subprocess is one-shot per
 chat.completions.create() call. A long-running ACP-style persistent session
 would be the upgrade path if per-call spawn latency becomes a bottleneck.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 1800.0  # 30 min — matches delegation.child_timeout ceiling

# Default args for the claude CLI. --dangerously-skip-permissions is NOT here;
# it is added only when HERMES_CLAUDE_SKIP_PERMS=true (default false = safe).
_DEFAULT_ARGS = [
    "--print",
    "--input-format", "stream-json",
    "--output-format", "stream-json",
    "--verbose",
]

# ===== Module-level shared PROCESS POOL =====
# All ClaudeStreamJsonClient instances share N claude processes (default 8)
# so concurrent delegate_task calls run in parallel instead of serializing on
# one subprocess stdin/stdout. Each slot is an independent claude process with
# its own session id. ponytail: list-of-slots with per-slot locks is enough at
# this concurrency level; swap to a real worker queue if pool_size grows large.
try:
    _pool_size = int(os.environ.get("HERMES_CLAUDE_POOL_SIZE", "8") or "8")
except (TypeError, ValueError):
    _pool_size = 8
if _pool_size < 1:
    _pool_size = 8

_shared_proc_pool: List[Optional[subprocess.Popen]] = [None] * _pool_size
_shared_proc_locks: List[threading.Lock] = [threading.Lock() for _ in range(_pool_size)]
_shared_proc_busy: List[bool] = [False] * _pool_size
_shared_slot_last_active: List[float] = [0.0] * _pool_size
_shared_slot_call_count: List[int] = [0] * _pool_size

# Aggregate counters kept for log continuity / back-compat diagnostics.
_shared_last_active: float = 0.0
_shared_call_count: int = 0


def _resolve_command() -> str:
    """Find the claude CLI binary robustly.

    Tries PATH lookup for claude / claude.exe / claude.bat, then the
    HERMES_CLAUDE_COMMAND env override. Raises FileNotFoundError if none found.
    """
    env_cmd = os.environ.get("HERMES_CLAUDE_COMMAND", "").strip()
    if env_cmd:
        resolved = shutil.which(env_cmd) or env_cmd
        return resolved

    for candidate in ("claude", "claude.exe", "claude.bat"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise FileNotFoundError(
        "Could not find the `claude` CLI on PATH. Install Claude Code, or set "
        "HERMES_CLAUDE_COMMAND to the binary path."
    )


def _resolve_args() -> List[str]:
    """Build the claude CLI args, honoring the skip-permissions env flag."""
    args = list(_DEFAULT_ARGS)

    skip_perms = os.environ.get("HERMES_CLAUDE_SKIP_PERMS", "").strip().lower()
    if skip_perms in ("1", "true", "yes"):
        args.append("--dangerously-skip-permissions")

    env_args = os.environ.get("HERMES_CLAUDE_ARGS", "").strip()
    if env_args:
        args.extend(shlex.split(env_args))

    return args


def _build_subprocess_env() -> dict:
    """Build env for the subprocess — inherit the parent so claude's auth
    (ANTHROPIC_API_KEY / config) and PATH propagate. Mirrors CopilotACPClient."""
    env = dict(os.environ)
    # Ensure HOME is set so claude finds ~/.claude
    env.setdefault("HOME", str(Path.home()))
    return env


def _format_messages_as_prompt(messages: List[dict]) -> str:
    """Flatten OpenAI-style messages into a single prompt text for claude.

    Claude Code gets ONE user turn per `--print` invocation. We fold the
    conversation transcript into a single text block. System messages become
    a leading directive; the last user message is the goal.
    """
    system_parts: List[str] = []
    transcript: List[str] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower()
        content = message.get("content")
        if isinstance(content, list):
            # OpenAI content-block list — extract text parts
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    text_parts.append(block)
            rendered = "\n".join(p for p in text_parts if p)
        elif isinstance(content, str):
            rendered = content
        else:
            continue
        if not rendered.strip():
            continue

        if role == "system":
            system_parts.append(rendered)
        else:
            label = {"user": "User", "assistant": "Assistant", "tool": "Tool"}.get(role, role.title())
            transcript.append(f"{label}:\n{rendered}")

    sections: List[str] = []
    if system_parts:
        sections.append("System instructions:\n" + "\n\n".join(system_parts))
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    if not sections:
        return ""
    return "\n\n".join(sections)


def _completion_to_stream_chunks(completion: SimpleNamespace) -> List[SimpleNamespace]:
    """Convert a one-shot response into OpenAI-style stream chunks (emulated streaming)."""
    choice = completion.choices[0]
    message = choice.message
    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=None,
        reasoning_content=getattr(message, "reasoning_content", None),
        reasoning=getattr(message, "reasoning", None),
    )
    data_chunk = SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=choice.finish_reason)],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(
        choices=[],
        model=completion.model,
        usage=completion.usage,
    )
    return [data_chunk, usage_chunk]


class _ClaudeChatCompletions:
    def __init__(self, client: "ClaudeStreamJsonClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeChatNamespace:
    def __init__(self, client: "ClaudeStreamJsonClient"):
        self.completions = _ClaudeChatCompletions(client)


class ClaudeStreamJsonClient:
    """OpenAI-compatible facade over the `claude` CLI stream-json subprocess."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[dict] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        acp_command: Optional[str] = None,
        acp_args: Optional[List[str]] = None,
        acp_cwd: Optional[str] = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-code"
        self.base_url = base_url or "acp://claude-code"
        self._default_headers = dict(default_headers or {})
        # command/args: explicit ctor args > env resolution
        self._command = command or acp_command or _resolve_command()
        self._args = list(args or acp_args or _resolve_args())
        self._cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ClaudeChatNamespace(self)
        self.is_closed = False
        self._active_process: Optional[subprocess.Popen] = None
        self._active_process_lock = threading.Lock()
        self._persistent_lock = self._active_process_lock  # alias for backward compat
        self._idle_timeout = 2592000.0  # 30 days — keep pool processes alive across reboots

    def close(self) -> None:
        """Close one-shot process only. Do NOT tear down the shared pool.
        Pool processes are kept alive for reuse. Recycled by idle_timeout or close_all().
        """
        proc: Optional[subprocess.Popen]
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        slot = getattr(self, "_active_slot", -1)
        if slot >= 0:
            _shared_proc_busy[slot] = False
            logger.debug("claude pool slot %d released (pid=%s, kept alive)",
                         slot, getattr(_shared_proc_pool[slot], "pid", "?"))

    def close_all(self) -> None:
        """Tear down ALL pool slots. Only call on gateway shutdown."""
        for i in range(_pool_size):
            with _shared_proc_locks[i]:
                _shared_proc_busy[i] = False
                p = _shared_proc_pool[i]
                _shared_proc_pool[i] = None
            if p is None:
                continue
            try:
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            logger.info("claude pool slot %d closed (pid=%s)", i, getattr(p, "pid", "?"))

    def _create_chat_completion(
        self,
        *,
        model: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        timeout: Optional[float] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        # Tools are accepted but ignored — Claude Code uses its native toolset.
        if isinstance(tools, list) and tools:
            logger.info(
                "ClaudeStreamJsonClient: %d tools passed but ignored (Claude Code uses native tools). "
                "Delegated goal must be self-contained.",
                len(tools),
            )

        prompt_text = _format_messages_as_prompt(messages or [])
        if not prompt_text.strip():
            prompt_text = "(empty prompt)"

        # Normalise timeout (run_agent may pass httpx.Timeout)
        if timeout is None:
            effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            effective_timeout = float(timeout)
        else:
            candidates = [getattr(timeout, attr, None) for attr in ("read", "write", "connect", "pool", "timeout")]
            numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
            effective_timeout = max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS

        result_text = self._run_prompt(prompt_text, timeout_seconds=effective_timeout)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=result_text,
            tool_calls=[],
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        completion = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "claude-code",
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion


    def _start_pool_proc(self, slot_index: int) -> subprocess.Popen:
        """Start a fresh claude process for the given pool slot."""
        try:
            proc = subprocess.Popen(
                [self._command] + self._args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self._cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start claude CLI command '{self._command}'. "
                "Install Claude Code or set HERMES_CLAUDE_COMMAND."
            ) from exc
        _shared_slot_last_active[slot_index] = time.monotonic()
        logger.info("claude pool slot %d proc started (pid=%s)", slot_index, proc.pid)
        return proc

    def _acquire_pool_slot(self):
        """Acquire an idle pool slot, starting the claude process if needed.

        Returns (slot_index, proc). If all slots are busy, waits for the
        first one to free up (with a generous timeout, then falls back to
        forcing the next slot). Marks the slot busy under its own lock so
        two callers never share the same proc's stdin/stdout.
        """
        # Fast path: find a free slot.
        wait_deadline = time.monotonic() + _DEFAULT_TIMEOUT_SECONDS
        while True:
            for i in range(_pool_size):
                if _shared_proc_busy[i]:
                    continue
                # Tentatively grab this slot's lock to claim it.
                with _shared_proc_locks[i]:
                    if _shared_proc_busy[i]:
                        continue  # lost a race, retry
                    proc = _shared_proc_pool[i]
                    # Recycle if dead, or if idle past the idle timeout.
                    if proc is not None and proc.poll() is not None:
                        logger.debug("claude pool slot %d proc exited (code=%s), recycling",
                                     i, proc.poll())
                        proc = None
                        _shared_proc_pool[i] = None
                    elif (proc is not None
                          and time.monotonic() - _shared_slot_last_active[i] > self._idle_timeout):
                        logger.debug("claude pool slot %d idle %.0fs, recycling",
                                     i, time.monotonic() - _shared_slot_last_active[i])
                        try:
                            proc.terminate()
                            proc.wait(timeout=5)
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                        proc = None
                        _shared_proc_pool[i] = None
                    if proc is None:
                        proc = self._start_pool_proc(i)
                        _shared_proc_pool[i] = proc
                    _shared_proc_busy[i] = True
                    # back-comat mirror to instance for any legacy reader
                    self._persistent_proc = proc
                    return i, proc
            # All slots busy — wait briefly and retry.
            if time.monotonic() > wait_deadline:
                # Last resort: force-claim slot 0 to avoid deadlock.
                logger.warning("claude pool all %d slots busy past timeout; forcing slot 0", _pool_size)
                i = 0
                with _shared_proc_locks[i]:
                    _shared_proc_busy[i] = True
                    proc = _shared_proc_pool[i]
                    if proc is None or proc.poll() is not None:
                        proc = self._start_pool_proc(i)
                        _shared_proc_pool[i] = proc
                    self._persistent_proc = proc
                    return i, proc
            time.sleep(0.05)

    def _release_pool_slot(self, slot_index: int) -> None:
        """Release a pool slot (keep the proc alive for warm reuse)."""
        with _shared_proc_locks[slot_index]:
            _shared_proc_busy[slot_index] = False

    def _ensure_persistent_process(self):
        """Back-compat shim — delegates to the pool (slot 0).

        Kept so any legacy caller still using the single-process entry point
        keeps working. Returns just the proc; slot tracking is not surfaced.
        """
        i, proc = self._acquire_pool_slot()
        # Release immediately; legacy callers don't hold the slot contract.
        # They get a live proc but are expected to be quick/single-threaded.
        self._release_pool_slot(i)
        return proc

    def _persistent_send_and_recv(self, proc, prompt_text, timeout_seconds, slot_index: int = 0):
        """Send a prompt to a pool slot's claude process and read the result."""
        global _shared_last_active
        request_line = json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            },
        })

        # Start stderr reader thread (daemon, shares stderr_tail list)
        stderr_tail: List[str] = []

        def _stderr_reader():
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        err_thread.start()

        # Write request (do NOT close stdin — persistent mode)
        proc.stdin.write(request_line + "\n")
        proc.stdin.flush()
        self._last_active = time.monotonic()
        _shared_slot_last_active[slot_index] = self._last_active

        result_text = None
        deadline = time.monotonic() + timeout_seconds
        for raw_line in proc.stdout:
            if time.monotonic() > deadline:
                logger.warning("claude pool slot %d recv timed out after %.0fs",
                               slot_index, timeout_seconds)
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            etype = event.get("type")
            if etype == "result":
                result_text = event.get("result", "")
                break

        now = time.monotonic()
        _shared_slot_last_active[slot_index] = now
        _shared_last_active = now  # aggregate for back-comat diagnostics
        self._last_active = now
        _shared_slot_call_count[slot_index] += 1
        self._call_count = _shared_slot_call_count[slot_index]

        if result_text is not None:
            logger.info("claude pool slot %d call #%d completed (pid=%s)",
                        slot_index, _shared_slot_call_count[slot_index], proc.pid)
            return result_text

        # Process may have died — clear this slot so next acquire restarts it.
        if proc.poll() is not None:
            logger.warning("claude pool slot %d proc exited (code=%s), will restart next call",
                           slot_index, proc.poll())
            with _shared_proc_locks[slot_index]:
                _shared_proc_pool[slot_index] = None
                self._persistent_proc = None

        stderr_excerpt = "\n".join(stderr_tail[-15:])
        raise RuntimeError(
            f"claude CLI pool slot {slot_index} did not produce result. "
            f"exit_code={proc.poll()} stderr_tail:\n{stderr_excerpt}"
        )

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float) -> str:
        """Acquire a pool slot, feed the prompt as one stream-json user message,
        read stdout JSON lines until a `result` event arrives, return its text.

        Parallel-safe: each concurrent caller gets its own slot's claude process
        (independent stdin/stdout), so 2+ delegate_task calls run in parallel.
        The slot is released in finally so it can never leak even on error.
        """
        slot_index = -1
        try:
            slot_index, proc = self._acquire_pool_slot()
            return self._persistent_send_and_recv(
                proc, prompt_text, timeout_seconds, slot_index=slot_index)
        except RuntimeError:
            # The slot's proc failed. Release it (so the dead proc can be
            # recycled on next acquire) and retry once on a fresh slot.
            if slot_index >= 0:
                # Force-clear the dead proc so the retry starts clean.
                with _shared_proc_locks[slot_index]:
                    dead = _shared_proc_pool[slot_index]
                    _shared_proc_pool[slot_index] = None
                    _shared_proc_busy[slot_index] = False
                if dead is not None:
                    try:
                        dead.kill()
                    except Exception:
                        pass
                self._persistent_proc = None
                slot_index = -1
            logger.warning("claude pool slot failed, retrying with fresh slot")
            slot_index, proc = self._acquire_pool_slot()
            return self._persistent_send_and_recv(
                proc, prompt_text, timeout_seconds, slot_index=slot_index)
        finally:
            if slot_index >= 0:
                self._release_pool_slot(slot_index)


# ===== Module-level singleton for persistent claude process reuse =====

_singleton_client: Optional["ClaudeStreamJsonClient"] = None
_singleton_lock = threading.Lock()


def get_shared_client(**kwargs) -> "ClaudeStreamJsonClient":
    """Get or create a shared ClaudeStreamJsonClient singleton.

    All delegate_task and dispatch_to_luban calls should use this instead of
    creating a new ClaudeStreamJsonClient() directly. The singleton keeps a
    POOL of persistent claude processes alive across calls, avoiding 30-40s
    cold starts, and lets 2+ concurrent delegate_task calls run in parallel
    (one per pool slot) instead of serializing on a single subprocess.
    """
    global _singleton_client
    with _singleton_lock:
        if _singleton_client is None:
            _singleton_client = ClaudeStreamJsonClient(**kwargs)
            logger.info("Created shared ClaudeStreamJsonClient singleton "
                        "(pool_size=%d, pids start on first use)", _pool_size)
        return _singleton_client


# Back-compat alias: legacy single-process name still resolves (points at
# pool[0] when populated). Old code that reads _shared_persistent_proc keeps
# working; nothing new should read it.
_shared_persistent_lock = _shared_proc_locks[0]  # legacy lock alias


def _shared_persistent_proc_get() -> Optional[subprocess.Popen]:
    """Legacy accessor — returns pool slot 0's proc, or None."""
    return _shared_proc_pool[0] if _shared_proc_pool else None


# Expose the legacy name as a module-level property-like object so bare
# `_shared_persistent_proc` reads (in any stale code path) don't NameError.
# It is a thin shim; writes go through _shared_persistent_proc_set().
class _SharedProcShim:
    def __bool__(self):
        return _shared_proc_pool and _shared_proc_pool[0] is not None

    def __repr__(self):
        return f"<_SharedProcShim pool[0]={_shared_proc_pool[0]!r}>"


_shared_persistent_proc = _SharedProcShim()
