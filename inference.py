import os
import re
import json
import time
from typing import Optional, List, Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

from env import AMLEnv
from models import Action, Observation

# Optional visualization (install networkx, matplotlib if you want graphs)
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

load_dotenv()

# ==================== Configuration ====================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
MAX_STEPS = 20
TEMPERATURE = 0.0          # deterministic, best for reproducible baseline
MAX_TOKENS = 300

# ==================== Helper Functions ====================
def build_prompt(observation: Observation) -> str:
    """
    Convert Observation into a structured prompt for the LLM.
    """
    if observation.current_transaction:
        tx_json = observation.current_transaction.model_dump_json(indent=2)
    else:
        tx_json = "None"

    prompt = (
        f"--- CURRENT STATE ---\n"
        f"Task ID: {observation.task_id}\n"
        f"Transactions already processed: {observation.transactions_processed}\n"
        f"Transactions you previously flagged: {observation.flags_made}\n"
        f"Info requests made: {observation.info_requested}\n"
        f"Done: {observation.done}\n\n"
        f"--- TRANSACTION TO REVIEW ---\n"
        f"{tx_json}\n\n"
        "Based on AML rules, decide the fate of this transaction. "
        "Respond with EXACTLY ONE of these commands and no other text:\n"
        "flag <transaction_id>\n"
        "request_info <transaction_id> \"<query>\"\n"
        "escalate\n"
        "skip"
    )
    return prompt

def parse_action(response_text: str) -> Action:
    """
    Robustly parse LLM output into an Action object.
    Scans from bottom up to catch final decision.
    """
    lines = response_text.strip().split("\n")
    # Read from bottom up to ignore any reasoning text above
    for line in reversed(lines):
        line = line.strip().lower()
        # Remove common prefixes like bullet points or dashes
        line = re.sub(r'^[-*:\s]+', '', line)

        if line.startswith("flag"):
            match = re.search(r'\d+', line)
            if match:
                return Action(type="flag", transaction_id=int(match.group()))
        elif line.startswith("request_info"):
            # Expect: request_info <id> "<query>"
            match = re.search(r"request_info\s+(\d+)\s+[\"'](.*?)[\"']", line)
            if match:
                return Action(
                    type="request_info",
                    transaction_id=int(match.group(1)),
                    query=match.group(2)
                )
        elif line.startswith("escalate"):
            return Action(type="escalate")
        elif line.startswith("skip"):
            return Action(type="skip")

    # Fallback if LLM goes off‑script
    return Action(type="skip")

# Optional: visualize transaction graph (not required for baseline)
def visualize_graph(graph, flagged_accounts=None, current_account=None):
    if not VISUALIZATION_AVAILABLE:
        return
    G = nx.DiGraph()
    for acc in graph.accounts:
        G.add_node(acc.id)
    for txn in graph.transactions:
        G.add_edge(txn.from_account, txn.to_account, amount=txn.amount)
    colors = []
    for node in G.nodes:
        if flagged_accounts and node in flagged_accounts:
            colors.append('red')
        elif current_account and node == current_account:
            colors.append('orange')
        else:
            colors.append('lightblue')
    nx.draw(G, with_labels=True, node_color=colors, node_size=800, font_size=10)
    plt.show()

# ==================== Main Task Runner ====================
def run_task(task_id: int) -> float:
    """
    Run a single AML task, controlling the environment with an LLM.
    Returns final score (0.0–1.0).
    """
    env = AMLEnv(task_id=task_id)
    obs = env.reset()
    total_reward = 0.0
    step = 0

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    while not obs.done and step < MAX_STEPS:
        prompt = build_prompt(obs)
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an elite Senior AML Investigator. Your goal is 100% accuracy in "
                            "catching money laundering while strictly avoiding false positives.\n\n"
                            "CRITICAL AML RULES:\n"
                            "1. HIGH RISK ENTITIES: You MUST `flag <id>` if the sender, receiver, or "
                            "description contains words like 'High-risk', 'Unknown', 'Shell', 'Offshore', "
                            "'Crypto', 'Darknet', or 'Sanctioned'.\n"
                            "2. STRUCTURING (SMURFING): You MUST `flag <id>` if the amount is suspiciously "
                            "close to the $10,000 reporting threshold (e.g., between $9,000 and $9,999).\n"
                            "3. LARGE UNEXPLAINED TRANSFERS: You MUST `flag <id>` for massive, round-number "
                            "transfers that lack a clear, legitimate business purpose.\n"
                            "4. SAFE TRANSACTIONS: You MUST `skip` everyday legitimate transactions. This "
                            "includes 'Legitimate Co', 'Salary', 'Client', 'Charity', 'Groceries', or regular "
                            "business expenses under $5,000.\n\n"
                            "Never explain your reasoning. Just output the command."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response = completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            response = "skip"

        action = parse_action(response)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        # Clean console output
        action_str = f"{action.type} {action.transaction_id if action.transaction_id else ''}".strip()
        print(f"Step {step:2d}: Action={action_str:<12} | Reward={reward:6.2f} | Done={done}")

    final_score = env._grade()
    print(f"Task {task_id} final score: {final_score:.2f}")
    return final_score

def main():
    scores = []
    for task_id in [1, 2, 3]:
        print(f"\n{'='*50}")
        print(f"Running Task {task_id}")
        print('='*50)
        score = run_task(task_id)
        scores.append(score)

    print("\n" + "="*50)
    print("Overall Baseline Scores:")
    print(f"Easy   (Task 1): {scores[0]:.2f}")
    print(f"Medium (Task 2): {scores[1]:.2f}")
    print(f"Hard   (Task 3): {scores[2]:.2f}")
    avg = sum(scores) / 3
    print(f"Average         : {avg:.2f}")
    if avg >= 0.90:
        print("🏆 SUCCESS: Target (>90%) Achieved!")
    print("="*50)

if __name__ == "__main__":
    main()