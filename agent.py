import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def print_box(title, data):
    print(f"\n=== {title} ===")
    print(json.dumps(data, indent=2))

def decide_action(observation):
    """
    Simple rule‑based agent.
    - Flags transactions that are clearly suspicious (large amount, unknown receiver, high‑risk location)
    - Escalates after processing all transactions (or when no more left)
    - Skips otherwise.
    """
    current_tx = observation.get("current_transaction")
    if current_tx is None:
        # No more transactions – escalate
        return {"action": {"type": "escalate"}}

    tx_id = current_tx["id"]
    amount = current_tx["amount"]
    receiver = current_tx["receiver"].lower()
    location = current_tx["location"].lower()

    # Suspicious conditions
    if amount > 100000:
        return {"action": {"type": "flag", "transaction_id": tx_id}}
    if "unknown" in receiver or "high‑risk" in location:
        return {"action": {"type": "flag", "transaction_id": tx_id}}
    # Structuring: amount between 9000 and 9999
    if 9000 <= amount <= 9999:
        return {"action": {"type": "flag", "transaction_id": tx_id}}

    # Otherwise safe
    return {"action": {"type": "skip"}}

def main():
    print("🚀 Starting OpenEnv Agent Workflow...\n")

    try:
        # 1. Health check
        print("Checking Server Health...")
        health_res = requests.get(f"{BASE_URL}/health")
        print_box("Health Status", health_res.json())

        # 2. Reset environment (task 1 – you can change task_id)
        print("Resetting Environment (Task 1)...")
        # Use query parameter as the server expects
        reset_res = requests.post(f"{BASE_URL}/reset?task_id=1")
        current_obs = reset_res.json()
        print_box("Initial Observation", current_obs)

        # 3. Interaction loop
        print("Starting Interaction Loop...")
        done = False
        step = 0
        max_steps = 20

        while not done and step < max_steps:
            step += 1
            print(f"\n--- Step {step} ---")

            # Decide action based on current observation
            action_payload = decide_action(current_obs.get("observation", {}))
            print(f"Action chosen: {action_payload}")

            # Send step request
            step_res = requests.post(f"{BASE_URL}/step", json=action_payload)
            step_data = step_res.json()
            print_box(f"Step {step} Response", step_data)

            # Extract observation and done flag
            current_obs = step_data  # The response already contains observation, reward, done, info
            done = step_data.get("done", False)

            # Optional: inspect state after first step
            if step == 1:
                print("\nInspecting internal State after Step 1...")
                state_res = requests.get(f"{BASE_URL}/state")
                print_box("Current Internal State", state_res.json())

            time.sleep(0.5)  # small delay to avoid overwhelming the server

        if done:
            print("\n✅ Episode finished successfully!")
            # Show final reward (if needed)
            if "reward" in step_data:
                print(f"Final reward (last step): {step_data['reward']}")
        else:
            print(f"\n⏸️ Reached max steps ({max_steps}) without finishing.")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the server. Make sure Uvicorn is running on port 8000!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()