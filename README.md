# AML Investigator Environment
# AML Investigator Environment

This OpenEnv environment simulates an anti-money laundering investigator reviewing transactions. It includes three tasks of increasing difficulty.

## Actions
- flag <transaction_id>
- request_info <transaction_id> "<query>"
- escalate
- skip

## Observations
- current_transaction
- transactions_processed
- flags_made
- info_requested
- done
- task_id

## Setup
Install dependencies:
pip install -r requirements.txt

Run baseline:
python inference.py

## Docker
Build:
docker build -t aml-investigator .

Run with API keys:
docker run -e HF_TOKEN=... -e API_BASE_URL=... -e MODEL_NAME=... -p 7860:7860 aml-investigator

## Hugging Face Space
Deployed at: [your space URL]