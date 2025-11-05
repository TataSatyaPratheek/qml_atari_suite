import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import mlflow

# Import ReplayBuffer and manual gradient functions
from replay_buffer import ReplayBuffer
import gradients

# Determine which gradient methods are handled by loss.backward()
# All others will be handled by gradients.py
AUTOGRAD_NATIVE_METHODS = ['backprop', 'adjoint', 'parameter-shift']

def train_agent(model, target_model, env, device, train_config, grad_config):
    """
    Generic training function for a DQN agent, integrated with MLflow
    and multiple gradient computation strategies.
    """
    # --- Unpack Training Config ---
    episodes = train_config['episodes']
    lr = train_config['lr']
    buffer_size = train_config['buffer_size']
    batch_size = train_config['batch_size']
    gamma = train_config['gamma']
    epsilon_start = train_config['epsilon_start']
    epsilon_end = train_config['epsilon_end']
    epsilon_decay = train_config['epsilon_decay']
    target_update = train_config['target_update']
    
    grad_method = grad_config['method']
    
    # --- Setup ---
    # Models are already on the correct device from run.py
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)
    loss_fn = nn.MSELoss()

    epsilon = epsilon_start
    episode_rewards = []
    
    mlflow.log_param("model_params", sum(p.numel() for p in model.parameters()))

    # --- Training Loop ---
    pbar = tqdm(range(episodes), desc=f"Training Run ({grad_method})")
    
    for episode in pbar:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        episode_loss = 0.0
        steps = 0
        done = False

        while not done:
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device) 
                    q_values = model(state_t)
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state, dtype=np.float32)

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps += 1

            if len(replay_buffer) < batch_size:
                continue
                
            # --- Perform one training step ---
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            states_t = torch.FloatTensor(states).to(device)
            actions_t = torch.LongTensor(actions).to(device)
            rewards_t = torch.FloatTensor(rewards).to(device)
            next_states_t = torch.FloatTensor(next_states).to(device)
            dones_t = torch.FloatTensor(dones).to(device)

            current_q = model(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_q = target_model(next_states_t).max(1)[0]
                target_q = rewards_t + gamma * next_q * (1 - dones_t)
            
            # --- GRADIENT COMPUTATION SWITCH ---
            
            # 1. Calculate loss
            loss = loss_fn(current_q, target_q)
            
            # 2. Zero out previous gradients
            optimizer.zero_grad()

            # 3. Compute gradients based on method
            if grad_method in AUTOGRAD_NATIVE_METHODS:
                # Use PyTorch's autograd. This works for:
                # - Classical models
                # - TensorCircuit (backprop, adjoint)
                # - PennyLane (adjoint, parameter-shift, backprop)
                loss.backward()
            else:
                # Use our manual gradient computation from gradients.py
                # This is for 'spsa', 'finite-diff', etc.
                gradients.compute_manual_gradient(
                    model, loss_fn, states_t, actions_t, target_q, grad_config
                )

            # 4. Apply gradients
            optimizer.step()
            # --- END OF GRADIENT SWITCH ---
            
            episode_loss += loss.item()

        # --- End of Episode ---
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)

        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())
        
        # --- MLflow Logging ---
        avg_loss = (episode_loss / steps) if steps > 0 else 0
        mlflow.log_metric("episode_reward", episode_reward, step=episode)
        mlflow.log_metric("avg_loss", avg_loss, step=episode)
        mlflow.log_metric("epsilon", epsilon, step=episode)

        # Update progress bar
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
        pbar.set_postfix({'Avg10': f'{avg_reward:.1f}', 'Eps': f'{epsilon:.3f}', 'Loss': f'{avg_loss:.4f}'})

    pbar.close()
    
    # --- Final Logging ---
    final_avg_reward = np.mean(episode_rewards[-10:])
    mlflow.log_metric("final_avg_reward_10", final_avg_reward)
    print(f"✓ Run complete. Final 10-episode avg reward: {final_avg_reward:.2f}")
    
    return episode_rewards