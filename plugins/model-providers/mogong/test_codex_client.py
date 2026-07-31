"""Independent unit test for CodexStreamJsonClient — does NOT import Hermes."""

import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from codex_client import CodexStreamJsonClient  # noqa: E402


def test_basic_pong() -> None:
    client = CodexStreamJsonClient()
    try:
        completion = client.chat.completions.create(
            model="mogong",
            messages=[{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}],
        )
        content = completion.choices[0].message.content
        assert content is not None, "completion.content is None"
        assert "PONG" in content, f"expected PONG, got: {content!r}"
        print(f"[PASS] basic_pong: content={content!r}")
    finally:
        client.close()


def test_streaming_emulated() -> None:
    client = CodexStreamJsonClient()
    try:
        chunks = client.chat.completions.create(
            model="mogong",
            messages=[{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}],
            stream=True,
        )
        collected = ""
        n = 0
        for chunk in chunks:
            n += 1
            if chunk.choices:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    collected += delta.content
        assert "PONG" in collected, f"stream missing PONG: {collected!r}"
        assert n >= 2, f"expected >=2 chunks, got {n}"
        print(f"[PASS] streaming: {n} chunks, content={collected!r}")
    finally:
        client.close()


def test_tools_ignored() -> None:
    client = CodexStreamJsonClient()
    try:
        completion = client.chat.completions.create(
            model="mogong",
            messages=[{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}],
            tools=[{"type": "function", "function": {"name": "x", "description": "y", "parameters": {}}}],
        )
        assert "PONG" in (completion.choices[0].message.content or "")
        print(f"[PASS] tools_ignored")
    finally:
        client.close()


def test_multi_message() -> None:
    client = CodexStreamJsonClient()
    try:
        completion = client.chat.completions.create(
            model="mogong",
            messages=[
                {"role": "system", "content": "You are a test echo. Always reply with exactly the word PONG."},
                {"role": "user", "content": "What is 2+2? Follow the system instruction."},
            ],
        )
        content = completion.choices[0].message.content or ""
        assert "PONG" in content, f"system not honored: {content!r}"
        print(f"[PASS] multi_message: content={content!r}")
    finally:
        client.close()


def _codex_available() -> bool:
    try:
        subprocess.run(["codex", "--version"], capture_output=True, timeout=15, check=False)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    if not _codex_available():
        print("[SKIP] codex CLI not on PATH", file=sys.stderr)
        return 77
    run_stream = "--stream" in sys.argv
    tests = [test_basic_pong, test_tools_ignored, test_multi_message]
    if run_stream:
        tests.insert(1, test_streaming_emulated)
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {t.__name__}: {exc!r}", file=sys.stderr)
    if failures:
        print(f"\n[RESULT] {failures} failed", file=sys.stderr)
        return 1
    print(f"\n[RESULT] all {len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
