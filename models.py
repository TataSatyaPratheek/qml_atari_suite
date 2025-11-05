import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

# --- Rich Logging Enhancement ---
from rich import print
# --- End Enhancement ---

# --- TensorCircuit Setup ---
# You must 'pip install tensorcircuit'
try:
    import tensorcircuit as tc
    tc.set_backend("pytorch")
    tc.set_dtype("float32")
    TENSORCIRCUIT_LOADED = True
except ImportError:
    print("[yellow]Warning:[default] TensorCircuit not installed. 'tensorcircuit' models will fail.")
    TENSORCIRCUIT_LOADED = False

# ============================================================================
# MODEL 1: Classical Baseline
# ============================================================================
class ClassicalDQN(nn.Module):
    """Classical Deep Q-Network baseline"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassicalDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ============================================================================
# MODEL 2: PennyLane Models
# ============================================================================

class PennyLaneBasicDQN(nn.Module):
    """Basic quantum approach - direct encoding without preprocessing"""
    
    def __init__(self, n_qubits, n_layers, n_actions, pl_device_name, grad_method_config):
        super().__init__()
        self.n_qubits = n_qubits
        
        diff_method = grad_method_config['method']
        if diff_method not in ['adjoint', 'parameter-shift', 'backprop']:
            diff_method = 'adjoint'

        dev = qml.device(pl_device_name, wires=n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        @qml.qnode(dev, interface='torch', diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')
            
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], 
                            weights[l, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.fc = nn.Linear(n_qubits, n_actions)

    def forward(self, x):
        # self.qlayer(x) runs on CPU, returns a CPU tensor
        q_out_cpu = self.qlayer(x)
        
        # --- FIX: Manually move the CPU tensor to the model's device ---
        q_out_on_device = q_out_cpu.to(self.fc.weight.device)
        
        # Now both tensors are on the same device (e.g., 'mps')
        return self.fc(q_out_on_device)


class PennyLaneHybridDQN(nn.Module):
    """Hybrid approach from paper: Classical encoder -> Quantum -> Classical decoder"""
    
    def __init__(self, input_dim, n_qubits, n_layers, n_actions, 
                 encoder_hidden_dim, decoder_hidden_dim, 
                 pl_device_name, grad_method_config):
        super().__init__()
        self.n_qubits = n_qubits

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, n_qubits),
            nn.Tanh()
        )
        
        diff_method = grad_method_config['method']
        if diff_method not in ['adjoint', 'parameter-shift', 'backprop']:
            diff_method = 'adjoint'

        dev = qml.device(pl_device_name, wires=n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        @qml.qnode(dev, interface='torch', diff_method=diff_method)
        def circuit(inputs, weights):
            for l in range(n_layers):
                qml.AngleEmbedding(inputs * np.pi, wires=range(self.n_qubits), rotation='Y')
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], 
                            weights[l, i, 2], wires=i)
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            qml.AngleEmbedding(inputs * np.pi, wires=range(self.n_qubits), rotation='Y')
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, n_actions)
        )

    def forward(self, x):
        # 'encoded' is on the correct device (e.g., 'mps')
        encoded = self.encoder(x)
        
        # 'q_out_cpu' is on 'cpu'
        q_out_cpu = self.qlayer(encoded)
        
        # --- FIX: Manually move the CPU tensor to the model's device ---
        # We get the device from the first layer of the decoder
        target_device = self.decoder[0].weight.device
        q_out_on_device = q_out_cpu.to(target_device)
        
        return self.decoder(q_out_on_device)

# ============================================================================
# MODEL 3: TensorCircuit Models
# (New implementation as requested)
# ============================================================================

if TENSORCIRCUIT_LOADED:
    
    # --- Keras-style Circuit for Basic DQN ---
    class BasicCircuitNG(tc.Circuit):
        def __init__(self, n_qubits, n_layers, **kwargs):
            super().__init__(n_qubits)
            self.n_layers = n_layers
            # Define weights as torch.nn.Parameters
            self.weights = nn.Parameter(
                torch.rand((n_layers, n_qubits, 3))
            )

        def forward(self, inputs):
            # inputs shape: (batch_size, n_qubits)
            # 2. AngleEmbedding
            for i in range(self.n_qubits):
                self.ry(i, theta=inputs[..., i])

            # 3. Variational Layers
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    self.rz(i, theta=self.weights[l, i, 0]) # phi
                    self.ry(i, theta=self.weights[l, i, 1]) # theta
                    self.rz(i, theta=self.weights[l, i, 2]) # omega
                for i in range(self.n_qubits - 1):
                    self.cz(i, i + 1)

            # 4. Measurements
            # Stack expectation values along axis=1 to get (batch_size, n_qubits)
            return tc.stack([self.expectation_ps(z=[i]) for i in range(self.n_qubits)], axis=1)

    # --- Main Model Class for Basic DQN ---
    class TensorCircuitBasicDQN(nn.Module):
        def __init__(self, n_qubits, n_layers, n_actions, **kwargs):
            super().__init__()
            self.q_layer = BasicCircuitNG(n_qubits, n_layers, **kwargs)
            self.fc = nn.Linear(n_qubits, n_actions)

        def forward(self, x):
            # We move input to CPU for tc-ng/mps compatibility
            q_out = self.q_layer(x.cpu())
            # We still take .real for MPS compatibility
            return self.fc(q_out.real.to(self.fc.weight.device))

    # --- Keras-style Circuit for Hybrid DQN ---
    class HybridCircuitNG(tc.Circuit):
        def __init__(self, n_qubits, n_layers, **kwargs):
            super().__init__(n_qubits)
            self.n_layers = n_layers
            self.weights = nn.Parameter(
                torch.rand((n_layers, n_qubits, 3))
            )

        def forward(self, inputs):
            # inputs shape: (batch_size, n_qubits)
            for l in range(self.n_layers):
                # Data re-uploading
                for i in range(self.n_qubits):
                    self.ry(i, theta=inputs[..., i] * np.pi)
                
                # Variational layer
                for i in range(self.n_qubits):
                    self.rz(i, theta=self.weights[l, i, 0]) # phi
                    self.ry(i, theta=self.weights[l, i, 1]) # theta
                    self.rz(i, theta=self.weights[l, i, 2]) # omega
                
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    self.cz(i, i + 1)
            
            # Final encoding
            for i in range(self.n_qubits):
                self.ry(i, theta=inputs[..., i] * np.pi)

            return tc.stack([self.expectation_ps(z=[i]) for i in range(self.n_qubits)], axis=1)

    # --- Main Model Class for Hybrid DQN ---
    class TensorCircuitHybridDQN(nn.Module):
        def __init__(self, input_dim, n_qubits, n_layers, n_actions, 
                     encoder_hidden_dim, decoder_hidden_dim, 
                     **kwargs):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(encoder_hidden_dim, n_qubits),
                nn.Tanh()
            )
            self.q_layer = HybridCircuitNG(n_qubits, n_layers, **kwargs)
            self.decoder = nn.Sequential(
                nn.Linear(n_qubits, decoder_hidden_dim),
                nn.ReLU(),
                nn.Linear(decoder_hidden_dim, n_actions)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            # Move input to CPU for tc-ng/mps compatibility
            q_out = self.q_layer(encoded.cpu())
            # We still take .real for MPS compatibility
            return self.decoder(q_out.real.to(self.decoder[0].weight.device))

else:
    # Create dummy classes if TensorCircuit is not installed
    class TensorCircuitBasicDQN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("TensorCircuit is not installed. Please 'pip install tensorcircuit' to use this model.")
            
    class TensorCircuitHybridDQN(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("TensorCircuit is not installed. Please 'pip install tensorcircuit' to use this model.")