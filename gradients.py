import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import time # <-- 1. IMPORT TIME

# ============================================================================
# This file contains MANUAL gradient computation techniques.
# These are called by train.py when grad_method is 'spsa' or 'finite-diff'.
# They compute gradients and set .grad attribute on model parameters.
# ============================================================================

# --- 2. IMPORT CUSTOM EXCEPTION ---
from exceptions import ExperimentTimeoutError

def compute_manual_gradient(model, loss_fn, states_t, actions_t, target_q, grad_config, start_time, timeout_sec): # <-- 3. ADD ARGS
    """
    Router for manual gradient computation.
    
    Args:
        model (nn.Module): The policy network.
        loss_fn (callable): The loss function (e.g., MSELoss).
        states_t (torch.Tensor): Batch of states.
        actions_t (torch.Tensor): Batch of actions.
        target_q (torch.Tensor): Target Q-values (detached).
        grad_config (dict): Configuration for the gradient method.
    """
    method = grad_config['method']
    
    if method == 'finite-diff':
        _compute_finite_diff(model, loss_fn, states_t, actions_t, target_q, grad_config, start_time, timeout_sec)
    elif method == 'spsa':
        _compute_spsa(model, loss_fn, states_t, actions_t, target_q, grad_config, start_time, timeout_sec)
    else:
        raise ValueError(f"Unknown manual gradient method: {method}")

def _compute_finite_diff(model, loss_fn, states_t, actions_t, target_q, grad_config, start_time, timeout_sec): # <-- 3. ADD ARGS
    """
    Computes gradient via the finite-difference rule.
    This is INCREDIBLY slow and intended for debugging only.
    """
    epsilon = grad_config.get('fd_epsilon', 1.0e-5)
    params = [p for p in model.parameters() if p.requires_grad]
    
    # 1. Compute original loss
    with torch.no_grad():
        current_q = model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        loss_orig = loss_fn(current_q, target_q)

    # 2. Iterate over every parameter
    for p in params:
        grad_tensor = torch.zeros_like(p.data)
        
        # Iterate over each element in the parameter tensor
        it = np.nditer(p.data.cpu().numpy(), flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            # --- 4. ADD TIMEOUT CHECK (INSIDE THE HOT LOOP) ---
            current_duration = time.time() - start_time
            if current_duration > timeout_sec:
                raise ExperimentTimeoutError(
                    f"Run exceeded smoke test timeout of {timeout_sec}s "
                    f"(current duration: {current_duration:.1f}s)"
                )
            # --- END TIMEOUT CHECK ---

            idx = it.multi_index
            orig_val = p.data[idx].item()
            
            # Perturb +
            p.data[idx] = orig_val + epsilon
            with torch.no_grad():
                q_plus = model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
                loss_plus = loss_fn(q_plus, target_q)
            
            # Compute gradient for this element
            grad = (loss_plus - loss_orig) / epsilon
            grad_tensor[idx] = grad
            
            # Restore original value
            p.data[idx] = orig_val
            
            it.iternext()
            
        # Set the computed gradient for the entire parameter tensor
        p.grad = grad_tensor.to(p.device)

def _compute_spsa(model, loss_fn, states_t, actions_t, target_q, grad_config, start_time, timeout_sec): # <-- 3. ADD ARGS
    """
    Computes gradient via Simultaneous Perturbation Stochastic Approximation (SPSA).
    This is a 2-sided implementation.
    """
    # SPSA hparams
    c_k = grad_config.get('spsa_c', 1e-3) # Perturbation size
    
    params = [p for p in model.parameters() if p.requires_grad]
    orig_params = [p.data.clone() for p in params]
    
    # 1. Generate Bernoulli perturbation vector (delta_k) for all params
    deltas = []
    for p in params:
        # delta_k is a tensor of +/- 1s with the same shape as p
        delta = (torch.randint(0, 2, p.shape, device=p.device) * 2 - 1).float()
        deltas.append(delta)

    # --- 4. ADD TIMEOUT CHECK (Before forward passes) ---
    current_duration = time.time() - start_time
    if current_duration > timeout_sec:
        raise ExperimentTimeoutError(
            f"Run exceeded smoke test timeout of {timeout_sec}s "
            f"(current duration: {current_duration:.1f}s)"
        )

    # 2. Compute loss_plus (theta + c_k * delta_k)
    for p, delta, orig in zip(params, deltas, orig_params):
        p.data = orig + c_k * delta
        
    with torch.no_grad():
        q_plus = model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        loss_plus = loss_fn(q_plus, target_q)

    # 3. Compute loss_minus (theta - c_k * delta_k)
    for p, delta, orig in zip(params, deltas, orig_params):
        p.data = orig - c_k * delta

    with torch.no_grad():
        q_minus = model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        loss_minus = loss_fn(q_minus, target_q)
        
    # 4. Compute gradient and set .grad attribute
    for p, delta, orig in zip(params, deltas, orig_params):
        # g_k = (y_plus - y_minus) / (2 * c_k * delta_k)
        grad_estimate = (loss_plus - loss_minus) / (2 * c_k * delta)
        p.grad = grad_estimate
        
        # Restore original parameters
        p.data = orig