# QML Atari Experiment Framework

**TLDR:** This repository is a scalable, MLflow-tracked framework to benchmark quantum reinforcement learning models. It directly compares PennyLane and TensorCircuit backends on the CartPole-v1 task. The framework is designed for "apples-to-apples" comparisons by supporting multiple gradient methods (`backprop`, `adjoint`, `parameter-shift`, `spsa`, `finite-diff`) across both QML libraries.

## Core Framework

This project is no longer a single script; it is a modular experiment runner:

  * **`config.yaml`**: Your main "control panel." Define all experiments, models, and parameters here.
  * **`run.py`**: The single entry point. It reads `config.yaml`, loops through all experiments, and handles errors/timeouts.
  * **`train.py`**: The core training loop. It automatically switches between native autograd (`loss.backward()`) and manual gradient calculation.
  * **`gradients.py`**: Implements manual gradient methods (SPSA, Finite-Difference) in a backend-agnostic way.
  * **`exceptions.py`**: A custom error class for handling timeouts.
  * **`models.py`**: The "QML lab." All model architectures (Classical, PennyLane, TensorCircuit) are defined here.
  * **`utils.py`**: Handles device setup (MPS/CUDA/CPU) and the model factory.
  * **`mlruns/`**: All results (metrics, parameters, status) are logged here.

## Smoke Test Results (5 Episodes @ 4 Qubits)

Our initial smoke test on an M1 Mac (MPS device) revealed critical performance and accuracy differences:

| Run Name | Backend | Gradient | Status | Time (s) | Final Avg Reward |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `smoke_test_classical` | Classical | `backprop` | **PASSED** | 1.18 | 18.00 |
| `smoke_test_pl_hybrid_adjoint` | PennyLane | `adjoint` | **PASSED** | 67.77 | 22.40 |
| `smoke_test_tc_hybrid_backprop`| TensorCircuit | `backprop` | **PASSED** | **1.86** | **22.80** |
| `smoke_test_pl_hybrid_spsa` | PennyLane | `spsa` | **PASSED** | 59.04 | 16.20 |
| `smoke_test_tc_hybrid_spsa` | TensorCircuit | `spsa` | **PASSED** | **2.42** | **18.80** |
| `smoke_test_pl_hybrid_param_shift`| PennyLane | `param-shift`| **FAILED** | \~0.5 | N/A |
| `smoke_test_pl_hybrid_finite_diff`| PennyLane | `finite-diff`| **TIMED OUT**| 150.2 | N/A |
| `smoke_test_tc_hybrid_finite_diff`| TensorCircuit | `finite-diff`| **TIMED OUT**| 150.0 | N/A |

## Key Findings (TLDR)

1.  **Time vs. Accuracy Tradeoff:** TensorCircuit is the clear winner. For native autograd, `tensorcircuit` (`backprop`, 1.86s) was **\~36x faster** than `pennylane` (`adjoint`, 67.77s) and achieved a *higher* reward (22.80 vs 22.40).

2.  **SPSA Speed:** The speed difference holds for manual methods. `tensorcircuit` (2.42s) was **\~24x faster** than `pennylane` (59.04s) and again achieved a higher reward (18.80 vs 16.20).

3.  **Bottleneck Identified:** The performance gap suggests PennyLane's `QNode` overhead is the primary bottleneck, not the `adjoint` method itself.

4.  **PennyLane Limitation:** The `parameter-shift` method **FAILED** due to a known PennyLane limitation with batched inputs ("broadcasted tapes"). This is a critical finding for our framework.

5.  **`finite-diff` is Unusable:** The `finite-diff` method is computationally infeasible and timed out for both backends, as expected.

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

All metrics, parameters, and run statuses (PASSED, FAILED, TIMED\_OUT) are logged to MLflow.

```bash
mlflow ui
```

## Possible Research Directions

1.  **Production Run (AWS)**: The framework is validated. The immediate next step is to run the "apples-to-apples" config (with `episodes: 500`) on AWS GPU instances to see if these time/accuracy tradeoffs hold at scale.

2.  **Migrate to `tensorcircuit-ng`**: We've confirmed `tensorcircuit` is faster. The next logical step is to migrate our `models.py` implementation from `tensorcircuit` to `tensorcircuit-ng` to see if its Keras-like API provides even better performance and a cleaner codebase.

3.  **Scale Up**: Add new, more complex model architectures to `models.py` or change the `env_name` in `config.yaml` to test on more difficult environments. The framework is built to handle it.