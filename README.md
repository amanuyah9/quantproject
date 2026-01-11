# quantproject

Event-driven macro impact modeling project.

## What this does
- Builds a clean, intraday event-impact dataset
- Labels volatility-adjusted market moves after macro events
- Trains a baseline classifier using walk-forward validation

## How to run
1. Install dependencies:
pip install -r requirements.txt
2. Build dataset:
python src/build_dataset.py
3. Train baseline model:
python src/train_baseline.py

Outputs are written to `outputs/`.
