import sys
import os
from env import AMLEnv
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from env import AMLEnv
from models import Action, Observation

# =====================================================================
# 1. FastAPI App Definition
# =====================================================================
app = FastAPI(
    title="AML Investigator Environment",
    description="Gymnasium-compliant OpenEnv environment for anti-money laundering investigations.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global environment instance
_active_env: Optional[AMLEnv] = None

# =====================================================================
# 2. Pydantic Models for Request/Response (Gymnasium Standard)
# =====================================================================
class StepRequest(BaseModel):
    action: Action

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    terminated: bool  # Gymnasium Standard: Did the game end naturally?
    truncated: bool   # Gymnasium Standard: Did the game end due to a time limit?
    done: bool        # Kept for backward compatibility with older UI dashboards
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: Observation
    info: Dict[str, Any]  # Gymnasium standard requires reset to also return info

class StateResponse(BaseModel):
    state: Dict[str, Any]

# =====================================================================
# 3. Standard OpenEnv Endpoints
# =====================================================================
@app.post("/reset", response_model=ResetResponse)
async def reset_endpoint(task_id: int = Query(1, ge=1, le=3)):
    """Reset the environment using the Gymnasium method."""
    global _active_env
    try:
        _active_env = AMLEnv(task_id=task_id)
        reset_result = _active_env.reset()
        
        # Smart Bridge: Handle both old Gym (1 item) and new Gymnasium (2 items)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
            
        return ResetResponse(observation=obs, info=info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResponse)
async def step_endpoint(request: StepRequest):
    """Execute an action using the Gymnasium 5-tuple method."""
    global _active_env
    if _active_env is None:
        raise HTTPException(status_code=400, detail="No active environment. Call /reset first.")
    
    try:
        step_result = _active_env.step(request.action)
        
        # Smart Bridge: Handle both old Gym (4 items) and new Gymnasium (5 items)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        elif len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False # Default fallback
        else:
            raise ValueError(f"Expected 4 or 5 return values from step(), got {len(step_result)}")

        return StepResponse(
            observation=obs, 
            reward=reward, 
            terminated=terminated, 
            truncated=truncated, 
            done=done, 
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=StateResponse)
async def state_endpoint():
    """Get the current internal state."""
    global _active_env
    if _active_env is None:
        raise HTTPException(status_code=400, detail="No active environment.")
    return StateResponse(state=_active_env.state())

# =====================================================================
# 4. Additional Custom Endpoints (for convenience)
# =====================================================================
@app.post("/custom_reset", response_model=ResetResponse)
async def custom_reset(task_id: int = Query(1, ge=1, le=3)):
    return await reset_endpoint(task_id)

@app.post("/custom_step", response_model=StepResponse)
async def custom_step(request: StepRequest):
    return await step_endpoint(request)

@app.get("/custom_state", response_model=StateResponse)
async def custom_state():
    return await state_endpoint()

# =====================================================================
# 5. Health Check & Documentation Redirect
# =====================================================================
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "AML Investigator Environment"}

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")

# =====================================================================
# 6. Entry Point
# =====================================================================
def main():
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True, log_level="info")

if __name__ == "__main__":
    main()