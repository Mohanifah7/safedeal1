# SafeDeal1 — Crypto AutoHedge Lab

A separate crypto research and paper-trading workspace. This repository is intentionally separate from the SafeDeal escrow product and from the `stock-tracker` US-equities project.

## Current status

- [x] Project scaffold and safe defaults
- [x] Public crypto market prices (no exchange credentials required)
- [x] Paper-trading simulator with a virtual balance
- [ ] AutoHedge agent integration — pending validation against the upstream package/API
- [ ] Persistent trade ledger and scheduled runs
- [ ] Deployment configuration

**This is not a live trading bot yet.** No real orders are sent and no private keys are required. The upstream AutoHedge project currently documents autonomous trading on Solana/Jupiter; its interface and execution behaviour must be verified before integration. Do not assume an environment variable alone can force its execution into paper mode.

## Run locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app reads public ticker data using CCXT and keeps simulated trades in the current app session. Restarting the app resets the demo ledger.

## Safety boundaries

- Paper trading only; no exchange API secrets or wallet private keys.
- No live order execution is implemented.
- Treat simulated performance as educational, not a prediction of future results.
- Keep this repository and its credentials separate from SafeDeal client/contractor escrow funds.
- Never commit private keys, seed phrases, access tokens, or production secrets.

## Upstream project

AutoHedge: https://github.com/The-Swarm-Corporation/AutoHedge
