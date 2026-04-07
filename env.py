from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# --- Local Project Imports ---
from models import Action, Observation
from tasks import get_task, Task

# =====================================================================
# 1. ENVIRONMENT DEFINITION (AMLEnv)
# =====================================================================
class AMLEnv:
    """
    Anti‑Money Laundering investigation environment.
    Implements the OpenEnv interface (step, reset, state, close).
    """
    def __init__(self, task_id: int = 1):
        self.task: Task = get_task(task_id)
        self.transactions = self.task.transactions
        self.ground_truth = self.task.ground_truth
        self.current_index = 0
        self.processed_ids: List[int] = []
        self.flags: List[int] = []
        self.info_requests: List[Dict[str, Any]] = []
        self.done = False
        self.steps_taken = 0
        self.max_steps = len(self.transactions) + 5
        self._final_score_given = False   # prevent double grading

    def reset(self, *args, **kwargs) -> Observation:
        """Reset the environment to start a new episode."""
        self.current_index = 0
        self.processed_ids = []
        self.flags = []
        self.info_requests = []
        self.done = False
        self.steps_taken = 0
        self._final_score_given = False
        return self._get_observation()

    def step(self, action: Action, *args, **kwargs) -> Tuple[Observation, float, bool, dict]:
        """
        Execute one action.
        Returns (observation, reward, done, info) as required by OpenEnv.
        """
        reward = 0.0
        info = {}

        if self.done:
            return self._get_observation(), 0.0, True, info

        # ---- Process action ----
        if action.type == "skip":
            self._move_next()
        elif action.type == "flag":
            if action.transaction_id is None:
                reward -= 0.1
            else:
                tx_id = action.transaction_id
                if tx_id in self.processed_ids:
                    reward -= 0.05   # already processed
                else:
                    self.flags.append(tx_id)
                    self._move_next()
                    if self.ground_truth.get(tx_id, False):
                        reward += 1.0   # correct flag
                    else:
                        reward -= 0.5   # false flag
        elif action.type == "request_info":
            if action.transaction_id is None or action.query is None:
                reward -= 0.1
            else:
                tx_id = action.transaction_id
                # Do NOT move to next transaction – info request is free
                self.info_requests.append({"id": tx_id, "query": action.query})
                if self.ground_truth.get(tx_id, False) and tx_id not in self.flags:
                    reward += 0.2   # good investigation
                else:
                    reward -= 0.1   # unnecessary request
        elif action.type == "escalate":
            self.done = True
            if not self._final_score_given:
                grade = self._grade()
                reward += grade
                self._final_score_given = True
        else:
            reward -= 0.2   # unknown action

        self.steps_taken += 1

        # ---- Check natural episode end (no escalate) ----
        if not self.done:
            if self.current_index >= len(self.transactions) or self.steps_taken >= self.max_steps:
                self.done = True
                if not self._final_score_given:
                    grade = self._grade()
                    reward += grade
                    self._final_score_given = True

        return self._get_observation(), reward, self.done, info

    def _move_next(self):
        """Advance to the next transaction (used by skip and flag)."""
        if self.current_index < len(self.transactions):
            tx = self.transactions[self.current_index]
            self.processed_ids.append(tx.id)
            self.current_index += 1

    def _get_observation(self) -> Observation:
        """Build the current observation from internal state."""
        current_tx = None
        if self.current_index < len(self.transactions):
            current_tx = self.transactions[self.current_index]
        return Observation(
            current_transaction=current_tx,
            transactions_processed=self.processed_ids.copy(),
            flags_made=self.flags.copy(),
            info_requested=self.info_requests.copy(),
            done=self.done,
            task_id=self.task.id
        )

    def _grade(self) -> float:
        """Compute final F1 score (0.0–1.0) based on flags vs ground truth."""
        true_positives = sum(1 for tx_id in self.flags if self.ground_truth.get(tx_id, False))
        false_positives = sum(1 for tx_id in self.flags if not self.ground_truth.get(tx_id, False))
        false_negatives = sum(1 for tx_id, truth in self.ground_truth.items() if truth and tx_id not in self.flags)

        if true_positives == 0:
            return 0.0
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def state(self) -> dict:
        """Return internal state (for debugging / inspection)."""
        return {
            "current_index": self.current_index,
            "processed_ids": self.processed_ids,
            "flags": self.flags,
            "info_requests": self.info_requests,
            "done": self.done,
            "steps_taken": self.steps_taken
        }

    # ---- OpenEnv required methods ----
    def close(self):
        """Clean up resources (none needed)."""
        pass

    async def reset_async(self, *args, **kwargs):
        """Async wrapper for reset."""
        return self.reset(*args, **kwargs)

    async def step_async(self, action, *args, **kwargs):
        """Async wrapper for step."""
        return self.step(action, *args, **kwargs)


# =====================================================================
# 2. FASTAPI SERVER ROUTES (OpenEnv HTTP API)
# =====================================================================
app = FastAPI(
    title="AML Investigator Environment",
    description="OpenEnv compliant environment for anti‑money laundering investigations.",
    version="1.0.0"
)

active_env: Optional[AMLEnv] = None

class StepRequest(BaseModel):
    action: Action

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: Observation

class StateResponse(BaseModel):
    state: Dict[str, Any]

@app.post("/reset", response_model=ResetResponse)
async def reset_endpoint(task_id: int = Query(1, ge=1, le=3)):
    """Reset the environment with a given task (1=easy, 2=medium, 3=hard)."""
    global active_env
    try:
        active_env = AMLEnv(task_id=task_id)
        obs = active_env.reset()
        return ResetResponse(observation=obs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResponse)
async def step_endpoint(request: StepRequest):
    """Send an action to the environment."""
    if active_env is None:
        raise HTTPException(status_code=400, detail="No active environment. Call /reset first.")
    try:
        obs, reward, done, info = active_env.step(request.action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=StateResponse)
async def get_state_endpoint():
    """Get the current internal state (debugging)."""
    if active_env is None:
        raise HTTPException(status_code=400, detail="No active environment.")
    return StateResponse(state=active_env.state())

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/grade")
async def get_grade():
    """Return the current F1 score (0.0–1.0) based on flags made."""
    if active_env is None:
        raise HTTPException(status_code=400, detail="No active environment.")
    return {"grade": active_env._grade()}