SDN controller utilities

Generators:
- VNF catalog generator
  python3 sdn_controller/dataset_generators/vnf_catalog_generator.py

Core reward/environment logic:
- sdn_controller/environment/sdn_env.py
- sdn_controller/environment/env_gym.py
- Reward function lives in sdn_controller/utilities/reward.py

Training:
- python3 sdn_controller/train.py
  - Configure policy in sdn_controller/configurations/config.yaml:
    - training.policy.use_custom: true/false
    - training.policy.net_arch: e.g., [256, 256]
    - training.policy.activation: relu | tanh | elu | leaky_relu
  - Custom PPO hyperparams (optional):
    - training.learning_rate, gamma, gae_lambda, clip_range, ent_coef, vf_coef
    - training.n_epochs, batch_size, max_grad_norm
  - Configure algorithm in sdn_controller/configurations/config.yaml:
    - training.algo: PPO | A2C
  - Plots:
    - paths.train_plot_path (eval reward during training)
  - Text logs:
    - paths.train_log_path

Plotting:
- python3 sdn_controller/plotter/plotter.py
  - Outputs:
    - paths.train_plot_path

Notes:
- Constants are defined in sdn_controller/configurations/config.yaml.
- load_config lives in sdn_controller/utilities/utils.py.
- Reward uses log2(1+SNIR) with profit + satisfaction only (no minimum-satisfaction penalty).
- User tiers are sampled each episode using probabilities in config.
- Action space is per-user: slice + PRBs + model per VNF.
- SNIR values are randomized every episode (train and eval) from config range.
