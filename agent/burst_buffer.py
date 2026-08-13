"""BurstBuffer — batch conversation turns before flushing to memory providers.

Sits between the agent loop and MemoryProvider.sync_turn(). Instead of
every completed turn hitting providers immediately, turns are accumulated
per (session, actor) key and flushed as a batch when either:

  * ``max_turns`` turns accumulate (full → flush now), or
  * ``quiet_ms`` elapses with no new turn (idle → flush).

Flush is dispatched through the MemoryManager's serialized background
worker (``_submit_background``), so turn N always lands before turn N+1
and a slow provider can never stall the turn path.

Scope: ONLY the per-turn ``sync_all`` conversation-persistence path is
buffered. Explicit memory-tool writes (``on_memory_write`` /
``notify_memory_tool_write``) and session-end extraction
(``on_session_end``) bypass the buffer entirely — those are user-driven
writes that must land immediately.

Disabled by default. Enable via ``memory.burst_buffer.enabled: true`` in
config.yaml or ``BURST_BUFFER_ENABLED=1`` in the environment.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnEntry:
    """One completed turn: the user input and assistant reply pair."""

    user_content: str
    assistant_content: str
    session_id: str = ""
    messages: Optional[List[Dict[str, Any]]] = None


@dataclass
class Burst:
    """A batch of buffered turns awaiting flush, keyed per session+actor."""

    session_id: str
    actor_id: str = ""
    turns: List[TurnEntry] = field(default_factory=list)
    timer: Optional[threading.Timer] = None


class BurstBuffer:
    """Accumulate completed turns and flush them to providers as a batch.

    The buffer keys on ``session_id`` (plus optional ``actor_id`` for
    gateway multi-user sessions) so concurrent sessions never coalesce.
    Thread-safe: ``add`` and ``flush_all`` may be called from any thread.

    The ``flush_fn`` callback receives the list of buffered turns and is
    expected to forward them to memory providers. It is invoked on a
    background thread — never the turn-completion path — so it must not
    raise (errors are logged and swallowed to protect the buffer).
    """

    def __init__(
        self,
        *,
        quiet_ms: int = 180_000,
        max_turns: int = 10,
        flush_fn: Optional[Callable[[str, str, List[TurnEntry]], None]] = None,
        on_error: Optional[Callable[[Exception, Burst], None]] = None,
    ) -> None:
        self.quiet_ms = quiet_ms
        self.max_turns = max_turns
        self.flush_fn = flush_fn
        self.on_error = on_error
        self._bursts: Dict[str, Burst] = {}
        self._lock = threading.Lock()
        self._closed = False

    def add(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        *,
        actor_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Buffer one completed turn. Flushes immediately if the batch fills."""
        if self._closed:
            return
        key = f"{session_id}\0{actor_id}"

        with self._lock:
            burst = self._bursts.get(key)
            if burst is not None:
                burst.turns.append(
                    TurnEntry(user_content, assistant_content, session_id, messages)
                )
                if burst.timer is not None:
                    burst.timer.cancel()
            else:
                burst = Burst(
                    session_id=session_id,
                    actor_id=actor_id,
                    turns=[TurnEntry(user_content, assistant_content, session_id, messages)],
                )
                self._bursts[key] = burst

            # Batch full → flush now, no timer needed.
            if len(burst.turns) >= self.max_turns:
                self._bursts.pop(key, None)
                turns = burst.turns
                burst.turns = []
                self._dispatch_flush(session_id, actor_id, turns)
                return

            # Arm the idle-timeout flush. Re-armed every add so the quiet
            # window resets on each new turn.
            burst.timer = threading.Timer(
                self.quiet_ms / 1000.0,
                self._flush_timeout,
                args=(key,),
            )
            burst.timer.daemon = True
            burst.timer.start()

    def _flush_timeout(self, key: str) -> None:
        """Idle timeout fired — flush that key's batch."""
        with self._lock:
            burst = self._bursts.pop(key, None)
        if burst is None:
            return
        if burst.timer is not None:
            burst.timer.cancel()
        self._dispatch_flush(burst.session_id, burst.actor_id, burst.turns)

    def _dispatch_flush(
        self, session_id: str, actor_id: str, turns: List[TurnEntry]
    ) -> None:
        """Invoke the flush callback, isolating the buffer from its failures."""
        if not turns:
            return
        try:
            if self.flush_fn is not None:
                self.flush_fn(session_id, actor_id, turns)
        except Exception as e:
            logger.error("BurstBuffer flush failed for session %s: %s", session_id, e)
            if self.on_error is not None:
                try:
                    self.on_error(
                        e, Burst(session_id=session_id, actor_id=actor_id, turns=turns)
                    )
                except Exception:
                    pass

    def flush_all(self) -> None:
        """Flush every pending batch. Call at session end / shutdown.

        Marks the buffer closed so late ``add`` calls after teardown are
        dropped rather than arming new timers that leak past shutdown.
        """
        with self._lock:
            self._closed = True
            bursts = list(self._bursts.values())
            self._bursts.clear()
        for burst in bursts:
            if burst.timer is not None:
                burst.timer.cancel()
            self._dispatch_flush(burst.session_id, burst.actor_id, burst.turns)

    def drain_pending(self) -> List[tuple]:
        """Pop every pending batch WITHOUT invoking flush_fn.

        Returns ``[(session_id, actor_id, turns), ...]``. The caller takes
        ownership of forwarding the turns — used by MemoryManager's
        ``on_session_end`` / ``shutdown_all`` where buffered turns must be
        replayed synchronously (inline) before the extraction/teardown that
        follows, rather than re-queued through the async flush callback
        (which would land them AFTER that extraction on the shared worker).

        Marks the buffer closed, like ``flush_all``.
        """
        with self._lock:
            self._closed = True
            bursts = list(self._bursts.values())
            self._bursts.clear()
        out = []
        for burst in bursts:
            if burst.timer is not None:
                burst.timer.cancel()
            out.append((burst.session_id, burst.actor_id, burst.turns))
        return out

    def pending_count(self) -> int:
        """Number of batches currently buffered (one per session+actor key)."""
        with self._lock:
            return len(self._bursts)

    def pending_turns(self) -> int:
        """Total turns across all buffered batches."""
        with self._lock:
            return sum(len(b.turns) for b in self._bursts.values())


def is_burst_buffer_enabled(mem_config: Optional[Dict[str, Any]]) -> bool:
    """Resolve the enable flag from config then environment.

    config.yaml::

        memory:
          burst_buffer:
            enabled: true
            quiet_ms: 180000
            max_turns: 10

    Env override (wins if set): ``BURST_BUFFER_ENABLED=1``.
    """
    env = os.environ.get("BURST_BUFFER_ENABLED")
    if env is not None:
        return env.strip() in ("1", "true", "yes", "on")
    if not mem_config:
        return False
    bb = mem_config.get("burst_buffer")
    if not isinstance(bb, dict):
        return False
    return bool(bb.get("enabled", False))


def burst_buffer_params(mem_config: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Read tunable params (quiet_ms, max_turns) from config with defaults."""
    if not mem_config:
        return {}
    bb = mem_config.get("burst_buffer")
    if not isinstance(bb, dict):
        return {}
    out: Dict[str, int] = {}
    if "quiet_ms" in bb:
        try:
            out["quiet_ms"] = int(bb["quiet_ms"])
        except (TypeError, ValueError):
            pass
    if "max_turns" in bb:
        try:
            out["max_turns"] = int(bb["max_turns"])
        except (TypeError, ValueError):
            pass
    return out


if __name__ == "__main__":
    # Self-check: buffer accumulates, fills, and flushes via callback.
    flushed: List[List[TurnEntry]] = []

    def capture(_sid: str, _actor: str, turns: List[TurnEntry]) -> None:
        flushed.append(turns)

    buf = BurstBuffer(quiet_ms=10_000_000, max_turns=3, flush_fn=capture)
    buf.add("s1", "u1", "a1")
    buf.add("s1", "u2", "a2")
    assert buf.pending_count() == 1 and buf.pending_turns() == 2, "should buffer, not flush"
    buf.add("s1", "u3", "a3")  # fills batch of 3 → immediate flush
    assert buf.pending_count() == 0, "full batch should have flushed"
    assert len(flushed) == 1 and len(flushed[0]) == 3, "one batch of 3"

    buf.add("s2", "u4", "a4")
    assert buf.pending_count() == 1, "separate session buffers separately"
    buf.flush_all()
    assert buf.pending_count() == 0 and len(flushed) == 2, "flush_all drains remaining"
    buf.add("s3", "u5", "a5")  # closed → dropped
    assert buf.pending_count() == 0, "closed buffer drops new turns"

    # Config resolution
    assert is_burst_buffer_enabled(None) is False
    assert is_burst_buffer_enabled({"burst_buffer": {"enabled": True}}) is True
    assert burst_buffer_params({"burst_buffer": {"quiet_ms": 5000, "max_turns": 7}}) == {
        "quiet_ms": 5000,
        "max_turns": 7,
    }
    print("BurstBuffer self-check passed")
