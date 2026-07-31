"""Claude Code provider profile (鲁班 transport).

Registers the `claude-code` provider via the model-provider plugin mechanism.
Like copilot-acp, this is an external subprocess transport — NOT a REST API.
The profile captures auth/endpoint metadata; the actual client construction
(ClaudeStreamJsonClient) is selected in auxiliary_client.py /
agent_runtime_helpers.py by the provider name.

api_mode="chat_completions" routes it through the OpenAI-compatible facade,
matching how CopilotACPClient plugs into run_agent. The ClaudeStreamJsonClient
exposes client.chat.completions.create() to satisfy that contract.
"""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCodeProfile(ProviderProfile):
    """Claude Code CLI — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is handled by the claude CLI subprocess itself."""
        return None


# ponytail: 主名用 "luban" 而非 "claude-code" — 避开 anthropic 插件把
# "claude-code" 当 alias 占用导致的 last-writer-wins 竞态(否则 alias 表
# 被 anthropic 覆盖,get_provider_profile("claude-code") 会返回 AnthropicProfile)。
luban = ClaudeCodeProfile(
    name="luban",
    aliases=("claude-code-cli", "claude_code_cli"),
    api_mode="chat_completions",  # stream-json subprocess uses chat_completions routing
    env_vars=(),  # auth handled by claude CLI's own config (~/.claude)
    base_url="acp://claude-code",  # internal scheme — not a real HTTP endpoint
    auth_type="external_process",
)

register_provider(luban)
