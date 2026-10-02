# CAP-MF: Empirically Grounded Interaction-Aware Scheduling for LLM Inference

CAP-MF is a research prototype for **energy-aware, interaction-aware scheduling of heterogeneous large language model inference requests**.

The project combines:

- Real vLLM latency and GPU-energy measurements
- Directional pairwise workload-interference profiles
- Request-length distributions derived from BurstGPT
- Exact, Greedy, and capacity-projected Mean-Field schedulers
- Sequential live replay on one-GPU and two-GPU vLLM configurations

The central question is:

> Can a capacity-projected Mean-Field approximation recover near-Exact interaction-aware assignments while providing better energy efficiency than a Greedy scheduler?

## Key idea

LLM requests differ in prompt length, output length, latency, and GPU-energy cost. Their behavior can also change when different request classes execute concurrently.

CAP-MF models this coupling using an empirically measured directional interaction coefficient:

```text
J_ij(c) = latency of class i when paired with class j on configuration c
          --------------------------------------------------------------- - 1
                     isolated latency of class i on configuration c
```

The scheduler uses these measurements to assign requests across feasible serving configurations under hard capacity constraints.

CAP-MF replaces a discontinuous overload penalty with an explicit projection onto the feasible probability region:

```text
For every request i:
    sum_c q_i(c) = 1

For every configuration c:
    sum_i q_i(c) <= capacity(c)

For every infeasible request/configuration pair:
    q_i(c) = 0
```

After convergence, a capacity-aware recovery step converts the probabilistic state into a valid discrete assignment.

## Research workflow

```text
vLLM TP1/TP2 profiling              BurstGPT trace
          |                               |
          v                               v
Measured energy, latency,        Trace-derived request classes
TTFT, and directional J_ij       and chronological windows
          |                               |
          +---------------+---------------+
                          |
                          v
                 Exact | Greedy | CAP-MF
                          |
                          v
               Capacity-feasible schedule
                          |
                          v
               Sequential live vLLM replay
                          |
                          v
         Energy | TTFT | latency | throughput
```

## Evaluated configuration

The reported prototype evaluation used:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Serving engine: vLLM
- Precision: FP16
- Hardware: 2 NVIDIA T4 GPUs
- TP1: one T4 GPU
- TP2: tensor parallelism across two T4 GPUs
- Workload classes: `SS`, `SM`, `SL`, `MS`, `MM`, `ML`, `LS`, `LM`, and `LL`

Exact software, CUDA, driver, and model-revision values should be captured from the retained experiment environment rather than inferred. See `capture_environment_manifest.py` in the publication bundle.

## Workload classes

Requests are classified by short, medium, or long input and output lengths:

| Input class | Output class | Bucket |
|---|---|---|
| Short | Short | `SS` |
| Short | Medium | `SM` |
| Short | Long | `SL` |
| Medium | Short | `MS` |
| Medium | Medium | `MM` |
| Medium | Long | `ML` |
| Long | Short | `LS` |
| Long | Medium | `LM` |
| Long | Long | `LL` |

The initial profiling targets were:

```text
Input tokens:
S = 128
M = 512
L = 1024

Output tokens:
S = 64
M = 128
L = 256
```

BurstGPT rows are mapped to the nearest measured operating point. Requests too far from the measured region are excluded instead of being assigned extrapolated performance values.

## Repository structure

```text
.
├── profiling/
│   ├── profile_vllm.py
│   ├── profile_interactions.py
│   └── build_empirical_profiles.py
├── scheduler/
│   ├── canonical.py
│   ├── empirical_profiles.py
│   ├── models.py
│   └── solvers.py
├── experiments/
│   ├── run_solver_comparison.py
│   └── run_empirical_replay.py
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── burstgpt_to_requests.py
├── sequential_burstgpt_replay.py
├── create_replicated_profiles.py
├── validate.py
└── README.md
```

Your checkout may contain only a subset of these files, depending on whether profiling, solver evaluation, live replay, and publication assets are stored together.

## Installation

A CUDA-capable Linux environment is required for live profiling and replay.

Example setup:

```bash
python -m pip install -U \
  vllm \
  torch \
  transformers \
  openai \
  requests \
  pandas \
  numpy \
  pulp
```

Verify the environment:

```bash
python -c "import torch, vllm; print(torch.__version__); print(vllm.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
```

## End-to-end workflow

### 1. Profile TP1

Start a TP1 vLLM server and run:

```bash
python -m profiling.profile_vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --config tp1 \
  --tp 1 \
  --capacity 8 \
  --gpus 0 \
  --output data/raw/tp1
```

Profile directional pairwise interactions:

```bash
python -m profiling.profile_interactions \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --config tp1 \
  --tp 1 \
  --gpus 0 \
  --single-summary data/raw/tp1/single_summary.csv \
  --output data/raw/tp1_interactions
```

### 2. Profile TP2

Stop TP1, start a TP2 vLLM server across both GPUs, and run:

```bash
python -m profiling.profile_vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --config tp2 \
  --tp 2 \
  --capacity 8 \
  --gpus 0,1 \
  --output data/raw/tp2
```

```bash
python -m profiling.profile_interactions \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --config tp2 \
  --tp 2 \
  --gpus 0,1 \
  --single-summary data/raw/tp2/single_summary.csv \
  --output data/raw/tp2_interactions
```

### 3. Build the processed empirical model

```bash
python -m profiling.build_empirical_profiles \
  --profile-dirs data/raw/tp1 data/raw/tp2 \
  --interaction-dirs data/raw/tp1_interactions data/raw/tp2_interactions \
  --output data/processed
```

The processed configuration table must use one common SLO per request bucket. SLO is a request-class property, not a configuration-specific property.

### 4. Validate the empirical model

```bash
python validate.py
```

Expected checks include:

```text
[PASS] empirical profile complete and finite
[PASS] Exact assignment valid
[PASS] Greedy assignment valid
[PASS] Mean-Field assignment valid
[PASS] Exact <= Greedy/Mean-Field objective
[PASS] Mean-Field simplex valid
All empirical checks passed.
```

### 5. Convert BurstGPT into scheduler requests

The raw BurstGPT file does not contain a `bucket` column. Convert it before using it with the schedulers:

```bash
python burstgpt_to_requests.py \
  --input /path/to/BurstGPT_1.csv \
  --max-rows 10000 \
  --max-profile-error 0.50 \
  --output-all data/processed/requests_all.csv \
  --output-covered data/processed/requests_covered.csv
```

The converted output retains trace metadata and adds:

```text
input_class
output_class
bucket
profile_input_tokens
profile_output_tokens
input_profile_error
output_profile_error
```

### 6. Run Exact, Greedy, and CAP-MF

```bash
python -m experiments.run_solver_comparison \
  --requests data/processed/requests_covered.csv \
  --limit 16 \
  --offset 0 \
  --lambda-interaction 1.0 \
  --tau 10.0 \
  --damping 0.10 \
  --output results/solver_results_n16_projected.csv
```

The result includes:

```text
objective
runtime_s
assignment_json
base_energy
interaction
gap_vs_exact_pct
iterations
converged
residual
simplex_error
capacity_violation
expected_occupancy
```

### 7. Run sequential live replay

On a two-GPU Kaggle instance, TP1 and TP2 cannot be maintained as independent live endpoints simultaneously because TP2 uses both GPUs. The replay runner therefore executes TP1 and TP2 sequentially:

```bash
python sequential_burstgpt_replay.py \
  --requests data/processed/requests_covered.csv \
  --solver-results results/solver_results_n16_projected.csv \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --limit 16 \
  --offset 0 \
  --repetitions 5 \
  --tp1-devices 0 \
  --tp2-devices 0,1 \
  --tp1-gpus 0 \
  --tp2-gpus 0,1 \
  --output-dir results/sequential_replay_n16_r5
```

The replay produces:

```text
prompt_audit.csv
replay_requests.csv
replay_windows.csv
gpu_samples.csv
replay_solver_repeats.csv
replay_solver_summary.csv
metadata.json
vllm_tp1.log
vllm_tp2.log
```

## Reported results

### Offline scheduling quality

Across ten chronological N=16 BurstGPT windows:

| Scheduler | Mean gap to Exact | Median gap | Maximum gap |
|---|---:|---:|---:|
| Exact | 0% | 0% | 0% |
| Greedy | 2.92798% | 2.81016% | 7.05268% |
| CAP-MF | **0.00073%** | **0%** | **0.00729%** |

CAP-MF diagnostics:

```text
Convergence rate:              100%
Mean iterations:               65.3
Maximum residual:              9.98e-7
Maximum capacity violation:    9.61e-11
Maximum simplex error:         0
```

### Five-repeat live replay

| Metric | Exact | Greedy | CAP-MF |
|---|---:|---:|---:|
| Success rate | 100% | 100% | 100% |
| Energy/request | 46.30 ± 0.74 J | 54.84 ± 0.73 J | **46.32 ± 1.08 J** |
| Mean latency | 3.300 ± 0.098 s | **3.217 ± 0.069 s** | 3.311 ± 0.084 s |
| P95 latency | 6.086 ± 0.241 s | **5.808 ± 0.158 s** | 6.079 ± 0.200 s |
| Virtual throughput | 2.589 ± 0.104 req/s | **2.681 ± 0.079 req/s** | 2.591 ± 0.087 req/s |

Relative to Greedy, CAP-MF produced:

```text
15.54% lower measured GPU energy per request
2.93% higher mean latency
3.34% lower virtual request throughput
```

Relative to Exact, CAP-MF remained within:

```text
0.05% measured GPU energy
0.12% virtual makespan
```

These results are specific to the retained model, hardware, profiles, workload window, and replay methodology.

## Understanding virtual makespan

TP1 and TP2 were measured sequentially on the two-GPU testbed. For each scheduler, virtual parallel makespan is defined as:

```text
max(TP1 replay-window duration, TP2 replay-window duration)
```

This metric estimates the completion time if the independently measured TP1 and TP2 configurations were available concurrently. It is **not** a measurement from a physically concurrent TP1-plus-TP2 cluster.

## N=32 scalability experiments

The N=32 experiment uses replicated logical configurations parameterized from measured TP1 and TP2 profiles:

```text
tp1_r0
tp1_r1
tp2_r0
tp2_r1
```

This is an **algorithmic scalability study**, not a physical four-replica deployment. The experiment assumes that replicas of the same measured configuration have identical profiles and no cross-replica interference.

Do not describe N=32 results as live four-replica hardware measurements unless such a system is actually deployed and measured.

## Reproducibility

Preserve the following artifacts for every reported result:

```text
Raw TP1 and TP2 profile CSVs
Raw interaction CSVs
Processed empirical profiles
BurstGPT conversion outputs
Solver result CSVs
Assignment JSON values
Convergence diagnostics
Replay request and window CSVs
GPU power samples
vLLM logs
Environment manifest
Resolved model revision
Plotting inputs and scripts
```

Capture the environment used for the final experiment:

```bash
python capture_environment_manifest.py
```

Never infer or manually guess software, CUDA, driver, or model-revision values.

## Limitations

- The physical testbed uses NVIDIA T4 GPUs.
- The evaluated model is relatively small.
- Only 40.26% of the sampled valid BurstGPT rows fell within the chosen empirical coverage threshold.
- BurstGPT does not contain original prompt text, so live replay reconstructs controlled prompts matching source token lengths.
- TP1 and TP2 replay is sequential on the two-GPU testbed.
- Pairwise interaction profiles may not capture all higher-order interference.
- The current objective is energy-dominated and should be normalized before broad multi-objective claims.
- N=32 uses replicated logical profiles unless additional physical replicas are deployed.



## Related work and data

- Woosuk Kwon et al., **Efficient Memory Management for Large Language Model Serving with PagedAttention**, SOSP 2023.
- Yuxin Wang et al., **BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems**, KDD 2025.
- Martin J. Wainwright and Michael I. Jordan, **Graphical Models, Exponential Families, and Variational Inference**, 2008.
- Agrim Bari, Parikshit Hegde, and Gustavo de Veciana, **Optimal Scheduling Algorithms for LLM Inference: Theory and Practice**, 2025.



## AI-use disclosure

Microsoft M365 Copilot assisted with software implementation, debugging, experimental workflow design, data analysis, plotting, documentation, and editing.

The repository maintainers are responsible for:

- Reviewing all generated code and text
- Validating calculations against retained artifacts
- Verifying every reference
- Ensuring that public content contains no confidential or restricted information
- Complying with applicable HPE AI-use, publication, security, and open-source policies

AI assistance should not be concealed or represented as independent authorship.

