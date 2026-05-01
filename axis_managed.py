import anthropic
import os
from pathlib import Path

AGENT_ID_FILE = Path(__file__).parent / ".axis_agent_id"

api_key = open(os.path.expanduser("~/.anthropic_key")).read().strip()
client = anthropic.Anthropic(api_key=api_key)

def get_or_create_agent():
    # Reuse existing agent if ID is saved
    if AGENT_ID_FILE.exists():
        agent_id = AGENT_ID_FILE.read_text().strip()
        try:
            agent = client.beta.agents.retrieve(agent_id, betas=["managed-agents-2026-04-01"])
            print(f"AXIS Agent Loaded:   {agent.id}")
            print(f"Name: {agent.name}")
            print("AXIS is ready.")
            return agent
        except Exception:
            print(f"Saved agent {agent_id} not found — creating a new one.")

    # Create fresh agent
    agent = client.beta.agents.create(
        name="AXIS",
        model="claude-sonnet-4-6",
        system="""You are AXIS — a personal executive AI system.

When you receive a task:
1. Think from multiple perspectives (Planner, Analyst, Critic, Optimizer, Executor)
2. Synthesize into one clear decision
3. Execute or recommend the best action

You are a partner, not a servant. Think before acting.""",
        tools=[{"type": "agent_toolset_20260401"}],
        betas=["managed-agents-2026-04-01"]
    )
    AGENT_ID_FILE.write_text(agent.id)
    print(f"AXIS Agent Created:  {agent.id}")
    print(f"Name: {agent.name}")
    print("AXIS is ready.")
    return agent

agent = get_or_create_agent()
