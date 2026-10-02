<<<<<<< HEAD
# CAP-MF-Empirically-Grounded-Interaction-Aware-Scheduling-for-Energy-Efficient-LLM-Inference
=======
# Zero-Synthetic Empirical Interaction-Aware LLM Scheduling

Headline empirical pipeline:

1. Serve the SAME model under each real configuration (e.g. TP1, TP2, TP4) using vLLM.
2. Run `profiling.profile_vllm` on each configuration. It measures latency and NVIDIA power; no synthetic latency/energy fallback exists.
3. Run `profiling.profile_interactions` for all 9x9 bucket pairs on each configuration. Directed interference is measured as latency degradation relative to measured solo latency.
4. Merge measured configurations with `profiling.build_empirical_profiles`.
5. Use a real/trace-derived request CSV, never generated random mixes for headline results.
6. Run Exact, Greedy, and empirical Mean-Field on the same measured E/L/J tables.
7. Replay resulting schedules against running vLLM configuration endpoints.

## Install

```bash
pip install -r requirements.txt
```

## Example: profile one REAL configuration

Start vLLM separately, for example on a machine with enough GPUs:

```bash
vllm serve MODEL --port 8000 --tensor-parallel-size 1
```

Then:

```bash
python -m profiling.profile_vllm \
  --model MODEL --config tp1 --tp 1 --capacity 8 --gpus 0 \
  --output data/raw/tp1

python -m profiling.profile_interactions \
  --model MODEL --config tp1 --tp 1 --gpus 0 \
  --single-summary data/raw/tp1/single_summary.csv \
  --output data/raw/tp1_interactions
```

Repeat for every REAL configuration. Example TP2 requires two actual visible GPUs and a vLLM server started with `--tensor-parallel-size 2`.

Merge:

```bash
python -m profiling.build_empirical_profiles \
  --profile-dirs data/raw/tp1 data/raw/tp2 \
  --interaction-dirs data/raw/tp1_interactions data/raw/tp2_interactions
```

## Real workload

```bash
python make_requests.py --input YOUR_REAL_TRACE.csv --bucket-column bucket
```

## Validate

```bash
python validate.py
```

## Solver comparison

```bash
python -m experiments.run_solver_comparison \
  --requests data/processed/requests.csv --limit 12
```

Exact is pairwise interaction-aware and therefore intentionally limited to small instances. Scale Mean-Field/Greedy to larger traces after Exact quality is established on small instances.

## Replay

Run endpoints for each measured config and map config names to URLs:

```bash
python -m experiments.run_empirical_replay \
  --requests data/processed/requests.csv \
  --model MODEL \
  --endpoint tp1=http://localhost:8001/v1 \
  --endpoint tp2=http://localhost:8002/v1
```

## Scientific rules

- No missing measured cell is synthesized.
- `EmpiricalProfiles` fails if E/L/J tables are incomplete.
- A single-GPU Colab session is a TP1 measurement, not TP2/TP4/TP8.
- Do not label a configuration TP2 unless the server actually ran with two GPUs and tensor parallel size 2.
- SLO uses a configured multiplier over measured solo latency in the starter implementation. For final paper results, replace this with your service's real SLO if available and state it explicitly.
- `nvidia-smi` power integration measures total visible serving-GPU board power over the workload window. With concurrent requests, `energy_j_per_request` is amortized window energy, not causal per-request attribution.
>>>>>>> 34de1b5 (added new)
