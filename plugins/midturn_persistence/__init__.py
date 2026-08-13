"""Mid-turn persistence plugin — hooks into agent lifecycle to persist streaming responses."""
from agent.midturn_persistence import MidTurnPersistenceHook

def init_plugin(agent):
    hook = MidTurnPersistenceHook(agent)
    hook.install()
    return hook
