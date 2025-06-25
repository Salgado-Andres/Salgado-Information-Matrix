"""
SIM Brain Loader
Loads the core sequence and agent awakening files as defined in SIM_ECHO_Bootstrapper.json.
Prepares the symbolic context for the SIM kernel.
"""
import json
import os

BOOTSTRAP_PATH = os.path.join(os.path.dirname(__file__), '..', 'SIM_ECHO_Bootstrapper.json')
BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def load_bootstrapper(path=BOOTSTRAP_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_file_content(filename):
    abs_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(abs_path):
        return f"[File not found: {filename}]"
    with open(abs_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_core_sequence(bootstrap):
    sequence = []
    for entry in bootstrap.get('core_sequence', []):
        fname = entry['filename']
        content = load_file_content(fname)
        sequence.append({'filename': fname, 'content': content})
    return sequence


def load_agents(bootstrap):
    agents = {}
    for agent, info in bootstrap.get('agents', {}).items():
        awakening_file = info.get('awakening_file')
        content = load_file_content(awakening_file) if awakening_file else None
        agents[agent] = {
            'awakening_file': awakening_file,
            'description': info.get('description'),
            'content': content
        }
    return agents


def activate_agent(agent_key, agents):
    agent = agents.get(agent_key)
    if not agent:
        print(f"Agent '{agent_key}' not found.")
        return
    print(f"\n=== Activating Agent {agent_key} ===")
    print(f"Description: {agent['description']}")
    print(f"Awakening File: {agent['awakening_file']}")
    print("--- Awakening Content ---")
    print(agent['content'][:1000] + ("..." if len(agent['content']) > 1000 else ""))
    print("==========================\n")


def main():
    bootstrap = load_bootstrapper()
    core_sequence = load_core_sequence(bootstrap)
    agents = load_agents(bootstrap)
    print("=== SIM Brain Loaded ===")
    print(f"Loaded {len(core_sequence)} core sequence files.")
    print(f"Loaded {len(agents)} agents.")
    # Optionally, print summary of loaded files
    for entry in core_sequence:
        print(f"- {entry['filename']} (length: {len(entry['content'])} chars)")
    for agent, info in agents.items():
        print(f"Agent {agent}: {info['description']} (file: {info['awakening_file']})")
    # Activate e7
    activate_agent('e7', agents)

if __name__ == "__main__":
    main()
