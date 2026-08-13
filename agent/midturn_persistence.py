#!/usr/bin/env python3
"""
C1: Mid-turn 持久化 — API 流式响应增量落盘

痛点：
    Hermes 已有 _flush_messages_to_session_db 在工具调用前后持久化，
    但 API 调用本身（LLM 正在生成的文本）在断连时全部丢失。
    如果 LLM 生成了 2000 token 的文本然后连接断了，这些文本没有被保存。

方案：
    在 Hermes 的流式回调链中注入一个 "增量落盘" 钩子。
    每收到 N 个 token 的 delta，就把当前累积的文本写入一个临时文件。
    断连后恢复时，新会话可以读取这个文件，把已生成的部分文本恢复回来。

    这是一个独立模块，通过 Hermes 插件系统加载，不修改核心代码。
    核心代码已有的 _flush_messages_to_session_db 在工具调用前后持久化，
    本模块补充的是 "API 正在流式生成文本时" 的增量持久化。

部署方式：
    1. 复制到 /mnt/i/.hermes/plugins/midturn-persistence/__init__.py
    2. 在 Hermes config.yaml 的 plugins 段启用
    3. 重启 Hermes

    或者作为独立 hook 脚本，通过 HERMES_LIFECYCLE_HOOKS 环境变量加载。

原理：
    Hermes 的 agent 有一个 stream_delta_callback，每收到一个流式 delta 就调用。
    本模块在这个回调链上插入一个 "写文件" 步骤：
    - 每 500ms 或每 2000 字符，把累积文本写入 /mnt/j/.../logs/midturn_buffer.md
    - turn 结束时删除临时文件（正常完成）
    - turn 异常中断时保留临时文件（断连恢复用）
    - 新 turn 开始时检查是否有遗留的临时文件

    同时，本模块还利用 Hermes 已有的 lifecycle hooks：
    - post_turn_finalize: turn 正常结束 → 清理临时文件
    - pre_api_request: API 请求前 → 记录请求参数（用于重放）
    - post_stream_delta: 流式 delta → 增量落盘（如果有这个 hook）
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("hermes.midturn_persistence")

# ============ 配置 ============
MIDTURN_BUFFER_DIR = Path(os.environ.get(
    "HERMES_MIDTURN_BUFFER_DIR",
    "/mnt/j/SimonApp/AI-Workspace/active/_midturn_buffers"
))
FLUSH_INTERVAL_MS = int(os.environ.get("HERMES_MIDTURN_FLUSH_MS", "500"))
FLUSH_CHAR_THRESHOLD = int(os.environ.get("HERMES_MIDTURN_FLUSH_CHARS", "2000"))
MAX_BUFFER_SIZE = int(os.environ.get("HERMES_MIDTURN_MAX_SIZE", "500000"))  # 500KB

class MidTurnBuffer:
    """增量落盘缓冲区 — 线程安全"""

    def __init__(self, session_id: str, turn_id: str):
        self.session_id = session_id
        self.turn_id = turn_id
        self.buffer = []
        self.total_chars = 0
        self.last_flush = time.monotonic()
        self.lock = threading.Lock()
        self.filepath = MIDTURN_BUFFER_DIR / f"{session_id}_{turn_id}.md"
        self.metadata_path = MIDTURN_BUFFER_DIR / f"{session_id}_{turn_id}.meta.json"
        self.closed = False

        # 确保目录存在
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # 写初始元数据
        self._write_meta({
            "session_id": session_id,
            "turn_id": turn_id,
            "start_time": datetime.now().isoformat(),
            "status": "streaming",
            "api_call_count": 0,
        })

    def append_delta(self, text: str):
        """追加一个流式 delta 到缓冲区"""
        if not text or self.closed:
            return

        with self.lock:
            self.buffer.append(text)
            self.total_chars += len(text)

            # 检查是否需要落盘
            now = time.monotonic()
            time_elapsed = (now - self.last_flush) * 1000
            should_flush = (
                time_elapsed >= FLUSH_INTERVAL_MS
                or self.total_chars >= FLUSH_CHAR_THRESHOLD
            )

            if should_flush:
                self._flush_unlocked()

    def _flush_unlocked(self):
        """把缓冲区内容写入文件（调用者持锁）"""
        if self.total_chars > MAX_BUFFER_SIZE:
            # 超过最大大小，截断前面的内容
            full_text = "".join(self.buffer)
            self.buffer = [full_text[-MAX_BUFFER_SIZE:]]
            self.total_chars = len(self.buffer[0])

        try:
            content = "".join(self.buffer)
            # 原子写
            tmp = self.filepath.with_suffix(".tmp")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(self.filepath)
            self.last_flush = time.monotonic()
        except Exception as e:
            logger.warning("Mid-turn buffer flush failed: %s", e)

    def _write_meta(self, meta: dict):
        """写元数据文件"""
        try:
            tmp = self.metadata_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.metadata_path)
        except Exception as e:
            logger.warning("Mid-turn meta write failed: %s", e)

    def finalize(self):
        """turn 正常结束 — 清理临时文件"""
        with self.lock:
            self.closed = True
            self._flush_unlocked()

        # 删除缓冲文件
        try:
            self.filepath.unlink(missing_ok=True)
            self.metadata_path.unlink(missing_ok=True)
        except Exception:
            pass

    def mark_interrupted(self):
        """标记为中断（断连）— 保留临时文件供恢复"""
        with self.lock:
            self._flush_unlocked()
            self._write_meta({
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "end_time": datetime.now().isoformat(),
                "status": "interrupted",
                "total_chars": self.total_chars,
            })
            self.closed = True

    def get_content(self) -> str:
        """获取当前缓冲区内容"""
        with self.lock:
            return "".join(self.buffer)


# ============ 全局缓冲区管理 ============
_active_buffers: dict[str, MidTurnBuffer] = {}
_buffers_lock = threading.Lock()

def get_or_create_buffer(session_id: str, turn_id: str) -> MidTurnBuffer:
    """获取或创建一个 mid-turn 缓冲区"""
    key = f"{session_id}_{turn_id}"
    with _buffers_lock:
        if key not in _active_buffers:
            _active_buffers[key] = MidTurnBuffer(session_id, turn_id)
        return _active_buffers[key]

def finalize_buffer(session_id: str, turn_id: str):
    """完成并清理缓冲区"""
    key = f"{session_id}_{turn_id}"
    with _buffers_lock:
        buf = _active_buffers.pop(key, None)
    if buf:
        buf.finalize()

def mark_buffer_interrupted(session_id: str, turn_id: str):
    """标记缓冲区为中断状态"""
    key = f"{session_id}_{turn_id}"
    with _buffers_lock:
        buf = _active_buffers.get(key)
    if buf:
        buf.mark_interrupted()

def find_interrupted_buffers() -> list[dict]:
    """查找所有中断的缓冲区（用于断连恢复）"""
    results = []
    if not MIDTURN_BUFFER_DIR.exists():
        return results

    for meta_file in MIDTURN_BUFFER_DIR.glob("*.meta.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("status") == "interrupted":
                buffer_file = meta_file.with_suffix("").with_suffix(".md")
                if buffer_file.exists():
                    meta["buffer_file"] = str(buffer_file)
                    meta["buffer_content"] = buffer_file.read_text(encoding="utf-8")[:5000]
                    results.append(meta)
        except Exception:
            pass

    return results

def cleanup_interrupted_buffer(session_id: str, turn_id: str):
    """清理一个已恢复的中断缓冲区"""
    prefix = f"{session_id}_{turn_id}"
    for f in MIDTURN_BUFFER_DIR.glob(f"{prefix}*"):
        try:
            f.unlink()
        except Exception:
            pass


# ============ Hermes 生命周期钩子 ============

def on_pre_api_request(**kwargs):
    """API 请求前 — 记录请求信息"""
    session_id = kwargs.get("session_id", "unknown")
    turn_id = kwargs.get("turn_id", "unknown")
    api_call_count = kwargs.get("api_call_count", 0)

    buf = get_or_create_buffer(session_id, turn_id)

    # 更新元数据
    buf._write_meta({
        "session_id": session_id,
        "turn_id": turn_id,
        "api_call_count": api_call_count,
        "model": kwargs.get("model", ""),
        "provider": kwargs.get("provider", ""),
        "timestamp": datetime.now().isoformat(),
        "status": "streaming",
    })

    # 记录请求参数（用于重放，但不含敏感信息）
    request_info = {
        "model": kwargs.get("model"),
        "provider": kwargs.get("provider"),
        "api_mode": kwargs.get("api_mode"),
        "message_count": kwargs.get("message_count"),
        "tool_count": kwargs.get("tool_count"),
        "retry_count": kwargs.get("retry_count"),
    }

    request_file = MIDTURN_BUFFER_DIR / f"{session_id}_{turn_id}.request.json"
    try:
        request_file.write_text(
            json.dumps(request_info, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass


def on_stream_delta(**kwargs):
    """流式 delta — 增量落盘"""
    session_id = kwargs.get("session_id", "unknown")
    turn_id = kwargs.get("turn_id", "unknown")
    text = kwargs.get("text", "")

    if text:
        buf = get_or_create_buffer(session_id, turn_id)
        buf.append_delta(text)


def on_turn_finalize(**kwargs):
    """turn 正常结束 — 清理缓冲区"""
    session_id = kwargs.get("session_id", "unknown")
    turn_id = kwargs.get("turn_id", "unknown")
    finalize_buffer(session_id, turn_id)


def on_turn_error(**kwargs):
    """turn 异常 — 保留缓冲区"""
    session_id = kwargs.get("session_id", "unknown")
    turn_id = kwargs.get("turn_id", "unknown")
    mark_buffer_interrupted(session_id, turn_id)


# ============ 恢复接口 ============

def get_recovery_info() -> list[dict]:
    """获取所有可恢复的中断 turn（供新会话调用）"""
    return find_interrupted_buffers()

def format_recovery_report() -> str:
    """格式化恢复报告（给 Agent 看）"""
    buffers = find_interrupted_buffers()
    if not buffers:
        return ""

    lines = ["## 🔄 检测到中断的 API 调用", ""]
    for buf in buffers:
        lines.append(f"### {buf.get('session_id', '?')}/{buf.get('turn_id', '?')}")
        lines.append(f"- 时间: {buf.get('end_time', buf.get('start_time', '?'))}")
        lines.append(f"- 模型: {buf.get('model', '?')}")
        lines.append(f"- 已生成: {buf.get('total_chars', 0)} 字符")
        preview = buf.get("buffer_content", "")[:500]
        if preview:
            lines.append(f"- 预览: {preview}...")
        lines.append("")

    return "\n".join(lines)


# ============ CLI 接口 ============

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mid-turn persistence manager")
    parser.add_argument("--check", action="store_true", help="Check for interrupted buffers")
    parser.add_argument("--clean", action="store_true", help="Clean all interrupted buffers")
    parser.add_argument("--report", action="store_true", help="Print recovery report")

    args = parser.parse_args()

    if args.check or args.report:
        buffers = find_interrupted_buffers()
        if not buffers:
            print("没有中断的 API 调用")
        else:
            print(f"发现 {len(buffers)} 个中断的 API 调用:")
            for buf in buffers:
                print(f"  - {buf.get('session_id')}/{buf.get('turn_id')}: {buf.get('total_chars', 0)} chars")

    if args.clean:
        if MIDTURN_BUFFER_DIR.exists():
            for f in MIDTURN_BUFFER_DIR.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            print("已清理所有中断缓冲区")


if __name__ == "__main__":
    main()
