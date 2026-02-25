# qed-rsa

QED-Nano with RSA (Recursive Self-Aggregation) on AIME 2025.

**QED-Nano** (`lm-provers/QED-Nano`, 4B params) is a math proof model fine-tuned from Qwen3-4B-Thinking.
**RSA** ([arxiv 2509.26626](https://arxiv.org/abs/2509.26626)) is a test-time scaling method that maintains a population of N solutions, subsamples K, asks the model to aggregate them into an improved solution, and repeats for T steps.

## Setup (RunPod)

```bash
# SSH into your RunPod pod (1x RTX 4090, 24GB VRAM)
git clone <this-repo> && cd qed-rsa
bash setup.sh
```

This installs dependencies and downloads QED-Nano weights to `/workspace/models/qed-nano/`.

## Running

### Option A: Server already running

If you launched the SGLang server separately:

```bash
bash scripts/run.sh 01_baseline
bash scripts/run.sh 02_rsa_basic
bash scripts/run.sh 03_rsa_verify
```

### Option B: Launch server separately

```bash
# Terminal 1: start the server
bash scripts/start_server.sh

# Terminal 2: run experiments (after server is ready)
bash scripts/run.sh 01_baseline
```

## Experiments

| Experiment | Description |
|---|---|
| `01_baseline` | QED-Nano pass@1 and pass@8 on AIME 2025 (30 problems). 8 independent samples per problem. |
| `02_rsa_basic` | RSA with N=8, K=3, T=5. Logs pass@1 and majority vote accuracy at each step t=0..5. |
| `03_rsa_verify` | RSA with self-verification + Boltzmann selection. Verifies candidates before aggregation and uses weighted subsampling. |

## Results

Results are saved to `/workspace/results/` as JSON files with per-problem breakdowns and aggregate metrics.

## Project Structure

```
qed-rsa/
├── setup.sh                      # Install deps, download model
├── requirements.txt
├── core/
│   ├── server.py                 # Launch/manage SGLang server
│   ├── generate.py               # Async batched generation via OpenAI client
│   ├── rsa.py                    # RSA loop implementation
│   ├── prompts.py                # Prompt templates (solve + aggregation)
│   └── eval.py                   # Answer extraction (\boxed{}) and evaluation
├── experiments/
│   ├── 01_baseline.py            # Baseline pass@1 and pass@8
│   ├── 02_rsa_basic.py           # RSA N=8, K=3, T=5
│   └── 03_rsa_verify.py          # RSA + self-verify + Boltzmann selection
├── data/
│   └── aime_2025.json            # AIME 2025 Part I + II (30 problems)
├── scripts/
│   ├── start_server.sh           # Launch SGLang server standalone
│   └── run.sh                    # Usage: bash scripts/run.sh <experiment>
└── README.md
```

## Configuration

Key parameters can be adjusted at the top of each experiment file:

- **N**: Population size (default 8)
- **K**: Subsample size for aggregation (default 3)
- **T**: Number of RSA steps (default 5)
- **MAX_TOKENS**: Max generation length (default 16384)
- **TEMPERATURE**: Sampling temperature (default 0.7)

## References

- [QED-Nano](https://huggingface.co/lm-provers/QED-Nano)
- [RSA: Recursive Self-Aggregation](https://arxiv.org/abs/2509.26626) / [Code](https://github.com/HyperPotatoNeo/RSA)
- [SGLang](https://github.com/sgl-project/sglang)
