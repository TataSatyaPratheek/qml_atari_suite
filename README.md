# QML Atari Experiment Framework

**TLDR:** This repository is a scalable, MLflow-tracked framework to benchmark quantum reinforcement learning models. It directly compares PennyLane and TensorCircuit backends on the CartPole-v1 task. The framework is designed for "apples-to-apples" comparisons by supporting multiple gradient methods (backprop, adjoint, parameter-shift, spsa, finite-diff) across both QML libraries.

## Core Framework

This project is no longer a single script; it is a modular experiment runner:

- `config.yaml`: Your main "control panel." Define all experiments, models, and parameters here.
- `run.py`: The single entry point. It reads `config.yaml`, loops through all experiments, and handles errors/timeouts.
- `train.py`: The core training loop. It automatically switches between native autograd (`loss.backward()`) and manual gradient calculation.
- `gradients.py`: Implements manual gradient methods (SPSA, Finite-Difference) in a backend-agnostic way.
- `models.py`: The "QML lab." All model architectures (Classical, PennyLane, TensorCircuit) are defined here.
- `utils.py`: Handles device setup (MPS/CUDA/CPU) and the model factory.
- `mlruns/`: All results (metrics, parameters, status) are logged here.

## Smoke Test Results (5 Episodes @ 4 Qubits)

Our initial smoke test on an M1 Mac (MPS device) revealed critical performance differences:

| Run Name | Backend | Gradient | Status | Time (s) |
|----------|---------|----------|--------|----------|
| smoke_test_classical | Classical | backprop | PASSED | 1.33 |
| smoke_test_pl_hybrid_adjoint | PennyLane | adjoint | PASSED | 56.95 |
| smoke_test_tc_hybrid_backprop | TensorCircuit | backprop | PASSED | 1.86 |
| smoke_test_pl_hybrid_spsa | PennyLane | spsa | PASSED | 64.43 |
| smoke_test_tc_hybrid_spsa | TensorCircuit | spsa | PASSED | 2.42 |
| smoke_test_pl_hybrid_param_shift | PennyLane | param-shift | FAILED | 0.70 |
| smoke_test_pl_hybrid_finite_diff | PennyLane | finite-diff | TIMED OUT | 150.2 |
| smoke_test_tc_hybrid_finite_diff | TensorCircuit | finite-diff | TIMED OUT | 150.0 |

## Key Findings (TLDR)

- **TensorCircuit is ~30x Faster**: For native autograd, tensorcircuit (backprop, 1.86s) is 30 times faster than pennylane (adjoint, 56.95s).
- **SPSA Speed**: The speed difference holds for manual methods. tensorcircuit (2.42s) is 26 times faster than pennylane (64.43s). This suggests PennyLane's QNode overhead is the main bottleneck.
- **PennyLane Limitation**: The parameter-shift method failed due to a known PennyLane limitation with batched inputs ("broadcasted tapes"). This is a critical finding for our framework.
- **finite-diff is Unusable**: The finite-diff method is computationally infeasible and timed out for both backends, as expected.

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Experiments

Modify `config.yaml` to define your experiments (e.g., set `episodes: 500` for a production run). Then, simply run:

```bash
python run.py
```

### View Results

All metrics, parameters, and run statuses (PASSED, FAILED, TIMED_OUT) are logged to MLflow.

```bash
mlflow ui
```

## Possible Research Directions

1. **Production Run (AWS)**: The framework is validated. The immediate next step is to run the "apples-to-apples" config (with `episodes: 500`) on AWS GPU instances to get statistically meaningful learning curves.

2. **Migrate to tensorcircuit-ng**: We've confirmed tensorcircuit is faster. The next logical step is to migrate our `models.py` implementation from tensorcircuit to tensorcircuit-ng (as we discussed) to see if its Keras-like API provides even better performance and a cleaner codebase.

3. **Deeper Gradient Analysis**: Investigate the trade-offs. Does PennyLane's slower adjoint method produce more stable gradients than TensorCircuit's fast backprop? A 500-episode run will answer this.

4. **Scale Up**: Add new, more complex model architectures to `models.py` or change the `env_name` in `config.yaml` to test on more difficult environments. The framework is built to handle it.
