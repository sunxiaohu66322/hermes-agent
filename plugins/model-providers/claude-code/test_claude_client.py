"""Independent unit test for ClaudeStreamJsonClient — does NOT import Hermes.
Runs outside the Hermes runtime to verify the client + claude CLI subprocess
contract in isolation. This is the regression baseline for the transport.

Usage:
    python3 plugins/model-providers/claude-code/test_claude_client.py
    python3 plugins/model-providers/claude-code/test_claude_client.py --stream
"""

import subprocess
import sys
import os

# Make claude_client.py importable without Hermes on the path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from claude_client import ClaudeStreamJsonClient  # noqa: E402


def test_basic_pong() -> None:
    """Single-turn: claude replies with exactly PONG, client returns it as content."""
    client = ClaudeStreamJsonClient()
    try:
        completion = client.chat.completions.create(
            model="claude-code",
            messages=[{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}],
        )
        content = completion.choices[0].message.content
        assert content is not None, "completion.content is None — no result event from claude"
        assert "PONG" in content, f"expected PONG in content, got: {content!r}"
        print(f"[PASS] basic_pong: content={content!r}")
    finally:
        client.close()


def test_streaming_emulated() -> None:
    """stream=True returns an iterable of OpenAI-style chunks with content."""
    client = ClaudeStreamJsonClient()
    try:
        chunks = client.chat.completions.create(
            model="claude-code",
            messages=[{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}],
            stream=True,
        )
        collected = ""
        n_chunks = 0
        for chunk in chunks:
            n_chunks += 1
            if chunk.choices:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    collected += delta.content
        assert "PONG" in collected, f"streamed content missing PONG: {collected!r}"
        assert n_chunks >= 2, f"expected >=2 emulated chunks, got {n_chunks}"
        print(f"[PASS] streaming: {n_chunks} chunks, content={collected!r}")
    finally:
        client.close()


def test_tools_ignored_not_error() -> None:
    """Passing tools does not raise — they are accepted and ignored."""
    client = ClaudeStreamJsonClient()
    try:
        completion = client.chat.completions.create(
            model="claude-code",
            messages=[{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}],
            tools=[{"type": "function", "function": {"name": "dummy", "description": "x", "parameters": {}}}],
        )
        assert "PONG" in (completion.choices[0].message.content or ""), "tools path lost PONG"
        print(f"[PASS] tools_ignored: completion returned despite tools arg")
    finally:
        client.close()


def test_multi_message_transcript() -> None:
    """Multi-message transcript folds into one prompt; system + user both honored."""
    client = ClaudeStreamJsonClient()
    try:
        completion = client.chat.completions.create(
            model="claude-code",
            messages=[
                {"role": "system", "content": "You are a test echo. Always reply with exactly the word PONG."},
                {"role": "user", "content": "What is 2+2? Reply per the system instruction."},
            ],
        )
        content = completion.choices[0].message.content or ""
        assert "PONG" in content, f"system instruction not honored, got: {content!r}"
        print(f"[PASS] multi_message: system+user folded, content={content!r}")
    finally:
        client.close()


def _claude_available() -> bool:
    """Check claude CLI is on PATH before running subprocess tests."""
    try:
        subprocess.run(["claude", "--version"], capture_output=True, timeout=15, check=False)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    if not _claude_available():
        print("[SKIP] claude CLI not on PATH — cannot run subprocess tests", file=sys.stderr)
        return 77  # skipped

    failures = 0
    run_stream = "--stream" in sys.argv
    tests = [test_basic_pong, test_tools_ignored_not_error, test_multi_message_transcript]
    if run_stream:
        tests.insert(1, test_streaming_emulated)

    for test in tests:
        try:
            test()
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc!r}", file=sys.stderr)

    if failures:
        print(f"\n[RESULT] {failures} test(s) failed", file=sys.stderr)
        return 1
    print(f"\n[RESULT] all {len(tests)} test(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
