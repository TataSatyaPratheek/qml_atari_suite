import torch
import numpy as np
import random
import mlflow
import pennylane as qml

# --- Rich Logging Enhancement ---
from rich import print
# --- End Enhancement ---

# Import all models from models.py
from models import (
    ClassicalDQN, 
    PennyLaneBasicDQN, 
    PennyLaneHybridDQN, 
    TensorCircuitBasicDQN,
    TensorCircuitHybridDQN
)

def set_seeds(seed=42):
    """Sets seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_device(preference="auto"):
    """
    Selects the best available device and corresponding QML backends.
    Returns:
        torch.device: The selected PyTorch device.
        str: The corresponding PennyLane device name.
        str: The corresponding TensorCircuit backend name.
    """
    pl_device_name = "default.qubit"
    tc_backend = "pytorch" # TensorCircuit will use PyTorch
    
    if preference == "auto":
        if torch.cuda.is_available():
            device_pref = "cuda"
        elif torch.backends.mps.is_available():
            device_pref = "mps"
        else:
            device_pref = "cpu"
    else:
        device_pref = preference

    if device_pref == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        pl_device_name = "lightning.gpu"
    elif device_pref == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        pl_device_name = "lightning.qubit" # lightning.gpu doesn't support MPS
    else:
        device = torch.device("cpu")
        pl_device_name = "lightning.qubit"
        
    # Check if lightning.gpu is actually available if selected
    if pl_device_name == "lightning.gpu":
        try:
            qml.device("lightning.gpu", wires=1)
        except (qml.DeviceError, ImportError):
            print("[yellow]Warning:[default] pennylane-lightning-gpu not found. Falling back to lightning.qubit.")
            pl_device_name = "lightning.qubit"
            
    print(f"[bold green]✓[/bold green] Using PyTorch Device: [bold cyan]{device}[/bold cyan]")
    print(f"[bold green]✓[/bold green] Using PennyLane Device: [bold cyan]{pl_device_name}[/bold cyan]")
    print(f"[bold green]✓[/bold green] Using TensorCircuit Backend: [bold cyan]{tc_backend}[/bold cyan]")
    return device, pl_device_name, tc_backend

def setup_mlflow(config):
    """Initializes MLflow tracking."""
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    try:
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        print(f"[bold green]✓[/bold green] MLflow experiment: [bold]{config['mlflow']['experiment_name']}[/bold]")
    except Exception as e:
        print(f"[yellow]Warning:[default] Could not set MLflow experiment. Using default. Error: {e}")
    print(f"[bold green]✓[/bold green] MLflow tracking URI: [bold]{config['mlflow']['tracking_uri']}[/bold]")

def get_model(model_config, env_state_dim, env_action_dim, 
              pl_device_name, tc_backend, grad_config):
    """
    Simple model factory based on config.
    """
    model_type = model_config['type']
    
    # Common args for all models
    args = {
        'input_dim': env_state_dim,
        'n_actions': env_action_dim,
        'pl_device_name': pl_device_name,
        'tc_backend': tc_backend,
        'grad_method_config': grad_config,
        **model_config # Add all model-specific params (n_qubits, etc.)
    }

    if model_type == "classical":
        model = ClassicalDQN(
            input_dim=args['input_dim'],
            hidden_dim=args['hidden_dim'],
            output_dim=args['n_actions']
        )
    elif model_type == "pennylane_basic":
        model = PennyLaneBasicDQN(
            n_qubits=args['n_qubits'],
            n_layers=args['n_layers'],
            n_actions=args['n_actions'],
            pl_device_name=args['pl_device_name'],
            grad_method_config=args['grad_method_config']
        )
    elif model_type == "pennylane_hybrid":
        model = PennyLaneHybridDQN(
            input_dim=args['input_dim'],
            n_qubits=args['n_qubits'],
            n_layers=args['n_layers'],
            n_actions=args['n_actions'],
            encoder_hidden_dim=args['encoder_hidden_dim'],
            decoder_hidden_dim=args['decoder_hidden_dim'],
            pl_device_name=args['pl_device_name'],
            grad_method_config=args['grad_method_config']
        )
    elif model_type == "tensorcircuit_basic":
        model = TensorCircuitBasicDQN(
            n_qubits=args['n_qubits'],
            n_layers=args['n_layers'],
            n_actions=args['n_actions'],
            grad_method_config=args['grad_method_config']
        )
    elif model_type == "tensorcircuit_hybrid":
        model = TensorCircuitHybridDQN(
            input_dim=args['input_dim'],
            n_qubits=args['n_qubits'],
            n_layers=args['n_layers'],
            n_actions=args['n_actions'],
            encoder_hidden_dim=args['encoder_hidden_dim'],
            decoder_hidden_dim=args['decoder_hidden_dim'],
            grad_method_config=args['grad_method_config']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    print(f"[bold green]✓[/bold green] Model [bold cyan]'{model_type}'[/bold cyan] initialized.")
    print(f"  → [dim]Parameters: {sum(p.numel() for p in model.parameters())}[/dim]")
    return model