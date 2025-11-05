import yaml
import gymnasium as gym
import mlflow
import time
from copy import deepcopy

# --- Rich Logging Enhancement ---
from rich import print
from rich.panel import Panel
# --- End Enhancement ---

import utils
import train

def merge_configs(global_cfg, exp_cfg):
    """Merges global config with experiment-specific overrides."""
    merged = deepcopy(global_cfg)
    for key, value in exp_cfg.items():
        if isinstance(value, dict) and key in merged:
            merged[key].update(value)
        else:
            merged[key] = value
    return merged

def run():
    # 1. Load Config
    print(Panel.fit("Loading configuration from [cyan]config.yaml[/cyan]", title="Setup"))
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Setup MLflow
    utils.setup_mlflow(config)
    
    # 3. Load Global Params
    global_training_config = config.get('global_training', {})
    global_device_config = config.get('global_device', {})
    env_name = config.get('environment', {}).get('env_name', 'CartPole-v1')

    # 4. Initialize Environment (to get dims)
    print(f"Initializing environment: [bold]{env_name}[/bold]")
    temp_env = gym.make(env_name)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    temp_env.close()

    # 5. Loop Through Experiments
    exp_header = f"Found [bold yellow]{len(config['experiments'])}[/bold yellow] experiments to run."
    print(Panel(exp_header, title="QML Experiment Runner", padding=(1, 2)))
    
    for i, exp_config in enumerate(config['experiments']):
        run_name = exp_config.get('run_name', f"experiment_{i+1}")
        print(Panel(f"Starting Experiment: [bold cyan]{run_name}[/bold cyan]", 
                    title=f"Run {i+1}/{len(config['experiments'])}"))

        with mlflow.start_run(run_name=run_name) as run:
            start_time = time.time()
            
            # --- A. Merge Configs ---
            train_cfg = merge_configs(global_training_config, exp_config.get('training', {}))
            grad_cfg = exp_config.get('gradient', {})
            model_cfg = exp_config.get('model', {})
            device_pref = exp_config.get('device', {}).get('preference', 
                                                       global_device_config.get('preference', 'auto'))
            
            # --- B. Log Configs to MLflow ---
            print(f"[dim]Logging parameters to MLflow...[/dim]")
            mlflow.log_param("run_name", run_name)
            mlflow.log_dict(model_cfg, "model_config.json")
            mlflow.log_dict(grad_cfg, "gradient_config.json")
            mlflow.log_dict(train_cfg, "training_config.json")
            mlflow.log_param("device_preference", device_pref)
            
            # --- C. Setup for this run ---
            seed = train_cfg.get('seed', 42)
            utils.set_seeds(seed)
            print(f"  [dim]Seeds set to {seed}[/dim]")
            
            torch_device, pl_device_name, tc_backend = utils.get_device(device_pref)
            mlflow.log_param("torch_device", str(torch_device))
            mlflow.log_param("pennylane_device", pl_device_name)
            mlflow.log_param("tensorcircuit_backend", tc_backend)
            
            run_env = gym.make(env_name)
            
            # --- D. Initialize Models ---
            policy_model = utils.get_model(
                model_cfg, state_dim, action_dim, 
                pl_device_name, tc_backend, grad_cfg
            ).to(torch_device)
            
            target_model = utils.get_model(
                model_cfg, state_dim, action_dim, 
                pl_device_name, tc_backend, grad_cfg
            ).to(torch_device)
            
            # --- E. Run Training ---
            print(f"Starting training for [bold]{run_name}[/bold] on [bold]{torch_device}[/bold]...")
            train.train_agent(
                model=policy_model,
                target_model=target_model,
                env=run_env,
                device=torch_device,
                train_config=train_cfg,
                grad_config=grad_cfg
            )
            
            run_env.close()
            
            # --- F. Log Final Time ---
            total_time = time.time() - start_time
            mlflow.log_metric("total_training_time_sec", total_time)
            print(f"[bold green]✓ Finished Experiment[/bold green]: [cyan]{run_name}[/cyan] in [yellow]{total_time:.2f}s[/yellow]")

    print(Panel("[bold green]✓ All experiments complete![/bold green]", 
                subtitle="Run [bold]`mlflow ui`[/bold] to view results.",
                padding=(1, 2)))

if __name__ == "__main__":
    run()