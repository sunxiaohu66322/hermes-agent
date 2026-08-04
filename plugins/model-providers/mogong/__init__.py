"""Mogong (Codex CLI) provider profile (墨工 transport).

Registers the `mogong` provider via the model-provider plugin mechanism.
Symmetric to the `luban` provider: external subprocess transport speaking
Codex's exec JSONL protocol. The CodexStreamJsonClient lives in this package;
client construction is selected in auxiliary_client.py / agent_runtime_helpers.py
by the provider name.
"""

from providers import register_provider
from providers.base import ProviderProfile


class MogongProfile(ProviderProfile):
    """Codex CLI — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        return None


# ponytail: 主名 "mogong" — 避开任何已注册 alias 冲突。
mogong = MogongProfile(
    name="mogong",
    aliases=("codex-cli", "mogong-cli"),
    api_mode="chat_completions",
    env_vars=(),  # auth from ~/.codex/auth.json
    base_url="acp://codex",
    auth_type="external_process",
)

register_provider(mogong)
