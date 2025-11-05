# QML Atari - Lean Experiment Framework (v2)

This repository implements a modular, MLflow-tracked framework for QML-RL experiments. It supports multiple backends (PennyLane, TensorCircuit) and multiple gradient strategies (backprop, adjoint, parameter-shift, SPSA, finite-diff) from a single configuration file.

## 🚀 Your Workflow

Your entire workflow is focused on three files:

1.  **`config.yaml`**: Define *what* to run. Add experiments, change models, and select gradient methods here.
2.  **`models.py`**: Define *how* models are built. Add new `nn.Module` classes for your new QML circuits (either PennyLane or TensorCircuit).
3.  **`gradients.py`**: Define new *manual* gradient logic. If you invent a new SPSA-like method, add it here.

### How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Experiments:**
    Open `config.yaml` and edit the `experiments:` list. You can add new runs, change `model: type` (e.g., `pennylane_basic`, `tensorcircuit_hybrid`) and `gradient: method` (e.g., `adjoint`, `backprop`, `spsa`).

3.  **Run All Experiments:**
    ```bash
    python run.py
    ```

4.  **View Results:**
    All metrics, parameters, and configs are logged to MLflow.
    ```bash
    mlflow ui
    ```
    Then open `http://127.0.0.1:5000` in your browser.

### 🛰️ How to Run on AWS (Multi-Instance)

This framework is built for parallel AWS runs.

1.  **Set up a central MLflow server** on a small EC2 instance and get its public IP (e.g., `34.22.11.4`).
2.  In your local `config.yaml`, change the `tracking_uri`:
    ```yaml
    mlflow:
      tracking_uri: "[http://34.22.11.4:5000](http://34.22.11.4:5000)"
    ```
3.  **Create different configs for each worker:**
    * `config_worker_1.yaml`: Contains experiments 1-5.
    * `config_worker_2.yaml`: Contains experiments 6-10.
4.  **Launch your AWS GPU instances:**
    * On **Worker 1**, upload `config_worker_1.yaml` (rename it to `config.yaml`) and the rest of the code. Run `python run.py`.
    * On **Worker 2**, upload `config_worker_2.yaml` (rename it to `config.yaml`) and the rest of the code. Run `python run.py`.

All instances will automatically log their results to your central MLflow server.