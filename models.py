from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class Transaction(BaseModel):
    id: int
    amount: float
    sender: str
    receiver: str
    timestamp: str
    location: str
    sender_occupation: Optional[str] = None
    receiver_relationship: Optional[str] = None

class Action(BaseModel):
    type: str = Field(..., description="Action type: 'flag', 'request_info', 'escalate', or 'skip'")
    transaction_id: Optional[int] = Field(None, description="ID of the transaction to act upon")
    query: Optional[str] = Field(None, description="Specific question about the transaction")

class Observation(BaseModel):
    current_transaction: Optional[Transaction] = None
    transactions_processed: List[int] = []
    flags_made: List[int] = []
    info_requested: List[Dict[str, Any]] = []
    done: bool = False
    task_id: int = 1

class Reward(BaseModel):
    value: float
    info: Dict[str, float]