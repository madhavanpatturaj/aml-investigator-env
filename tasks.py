from models import Transaction
from typing import List, Dict

class Task:
    def __init__(self, id: int, name: str, description: str,
                 transactions: List[Transaction], ground_truth: Dict[int, bool]):
        self.id = id
        self.name = name
        self.description = description
        self.transactions = transactions
        self.ground_truth = ground_truth

def get_task(task_id: int):
    if task_id == 1:
        # Easy: single suspicious transaction
        tx = Transaction(
            id=1, amount=500000, sender="John Doe", receiver="Unknown Corp",
            timestamp="2024-01-15T10:00:00", location="High‑risk country",
            sender_occupation="Unemployed", receiver_relationship="Unknown"
        )
        ground_truth = {1: True}
        return Task(1, "Easy", "Single suspicious transaction", [tx], ground_truth)

    elif task_id == 2:
        # Medium: structuring – three transactions just under $10k
        txs = [
            Transaction(id=1, amount=9500, sender="Alice", receiver="Bob",
                        timestamp="2024-01-15T09:00:00", location="USA"),
            Transaction(id=2, amount=9800, sender="Alice", receiver="Bob",
                        timestamp="2024-01-15T09:15:00", location="USA"),
            Transaction(id=3, amount=9700, sender="Alice", receiver="Bob",
                        timestamp="2024-01-15T09:30:00", location="USA"),
        ]
        ground_truth = {1: True, 2: True, 3: True}
        return Task(2, "Medium", "Structuring – multiple small transactions", txs, ground_truth)

    elif task_id == 3:
        # Hard: mixture with false positives, need to request info
        txs = [
            Transaction(id=1, amount=15000, sender="Charlie", receiver="Legitimate Co",
                        timestamp="2024-01-15T11:00:00", location="USA",
                        sender_occupation="Business Owner", receiver_relationship="Client"),
            Transaction(id=2, amount=20000, sender="Charlie", receiver="Foreign Entity",
                        timestamp="2024-01-15T11:05:00", location="High‑risk country"),
            Transaction(id=3, amount=500, sender="Charlie", receiver="Charity",
                        timestamp="2024-01-15T11:10:00", location="USA",
                        sender_occupation="Business Owner", receiver_relationship="Donation"),
        ]
        ground_truth = {2: True}
        return Task(3, "Hard", "Complex mixture with false positives", txs, ground_truth)

    else:
        raise ValueError("Invalid task id")