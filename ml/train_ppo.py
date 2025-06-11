import os
import argparse
import torch
import numpy as np
import yaml
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import wandb

from gym_env import RepDetectionEnv


class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_logging()

    def setup_logging(self):
        """Initialize logging with wandb if enabled"""
        if self.config.use_wandb:
            wandb.init(
                project="rep-detection-ppo",
                config=self.config.__dict__,
                name=f"ppo_{self.config.subject}_{self.config.exercise}",
            )

    def make_env(self, rank=0):
        """Create and wrap environment"""

        def _init():
            env = RepDetectionEnv(
                subject=self.config.subject,
                exercise=self.config.exercise,
                dataset_root=self.config.dataset_root,
                c=self.config.c,
                m=self.config.m,
                tol=self.config.tol,
            )
            env = Monitor(env)
            return env

        return _init

    def create_vec_env(self):
        """Create vectorized environment"""
        if self.config.n_envs == 1:
            return DummyVecEnv([self.make_env()])
        else:
            return SubprocVecEnv([self.make_env(i) for i in range(self.config.n_envs)])

    def create_model(self, env):
        """Create PPO model with custom policy network"""

        # PPO hyperparameters
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=dict(
                net_arch=[self.config.hidden_dim] * self.config.n_layers,
                activation_fn=torch.nn.ReLU,
            ),
            tensorboard_log=f"./tensorboard_logs/" if self.config.tensorboard else None,
            verbose=1,
        )

        # Optionally initialize from a supervised learning checkpoint
        if getattr(self.config, "supervised_path", None):
            chk = self.config.supervised_path
            if os.path.exists(chk):
                try:
                    state_dict = torch.load(chk, map_location="cpu")
                    policy_state = model.policy.state_dict()
                    for (p_name, p_tensor), (_, s_tensor) in zip(
                        policy_state.items(), state_dict.items()
                    ):
                        if p_tensor.shape == s_tensor.shape:
                            policy_state[p_name] = s_tensor
                    model.policy.load_state_dict(policy_state, strict=False)
                    print(f"Loaded supervised weights from {chk}")
                except Exception as e:
                    print(f"Failed to load supervised weights: {e}")
            else:
                print(f"Supervised checkpoint not found: {chk}")

        return model

    def create_callbacks(self, eval_env):
        """Create training callbacks"""
        callbacks = []

        # Evaluation callback
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=self.config.save_path,
                log_path=self.config.save_path,
                eval_freq=self.config.eval_freq,
                deterministic=getattr(self.config, "eval_deterministic", False),
                render=False,
                n_eval_episodes=self.config.n_eval_episodes,
            )
            callbacks.append(eval_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.checkpoint_freq,
            save_path=os.path.join(self.config.save_path, "checkpoints"),
            name_prefix="ppo_rep_detection",
        )
        callbacks.append(checkpoint_callback)

        return callbacks

    def train(self):
        """Main training loop"""
        print(f"Starting PPO training for {self.config.subject}/{self.config.exercise}")
        print(f"Training for {self.config.total_timesteps} timesteps")

        # Create environments
        train_env = self.create_vec_env()

        # Create evaluation environment if specified
        eval_env = None
        if self.config.eval_subject and self.config.eval_exercise:
            eval_env = DummyVecEnv(
                [
                    lambda: Monitor(
                        RepDetectionEnv(
                            subject=self.config.eval_subject,
                            exercise=self.config.eval_exercise,
                            dataset_root=self.config.dataset_root,
                            c=self.config.c,
                            m=self.config.m,
                            tol=self.config.tol,
                        )
                    )
                ]
            )

        # Create model
        model = self.create_model(train_env)

        # Create callbacks
        callbacks = self.create_callbacks(eval_env)

        # Load existing model if specified
        if self.config.load_path:
            print(f"Loading model from {self.config.load_path}")
            model = PPO.load(self.config.load_path, env=train_env)

        # Train the model
        try:
            model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                log_interval=self.config.log_interval,
                tb_log_name="ppo_rep_detection",
                reset_num_timesteps=not self.config.load_path,
            )

            # Save final model
            final_path = os.path.join(self.config.save_path, "final_model")
            model.save(final_path)
            print(f"Final model saved to {final_path}")

        except KeyboardInterrupt:
            print("Training interrupted by user")

        finally:
            # Clean up
            train_env.close()
            if eval_env:
                eval_env.close()
            if self.config.use_wandb:
                wandb.finish()

    def evaluate(self, model_path, n_episodes=10):
        """Evaluate a trained model"""
        print(f"Evaluating model from {model_path}")

        # Create environment
        env = RepDetectionEnv(
            subject=self.config.eval_subject or self.config.subject,
            exercise=self.config.eval_exercise or self.config.exercise,
            dataset_root=self.config.dataset_root,
            c=self.config.c,
            m=self.config.m,
            tol=self.config.tol,
        )

        # Load model
        model = PPO.load(model_path)

        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        replay_data = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            # Initialize episode replay data if saving replays
            episode_replay = None
            if hasattr(self.config, "save_replay") and self.config.save_replay:
                episode_replay = {
                    "episode_number": episode + 1,
                    "subject": self.config.eval_subject or self.config.subject,
                    "exercise": self.config.eval_exercise or self.config.exercise,
                    "model_path": model_path,
                    "env_config": {
                        "c": self.config.c,
                        "m": self.config.m,
                        "tol": self.config.tol,
                        "dataset_root": self.config.dataset_root,
                    },
                    "initial_info": info,
                    "steps": [],
                }

            while not done:
                action, _ = model.predict(
                    obs, deterministic=getattr(self.config, "eval_deterministic", False)
                )
                new_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

                # Save step data if recording replay
                if episode_replay is not None:
                    step_data = {
                        "timestep": episode_length,
                        "observation": obs.tolist() if hasattr(obs, "tolist") else obs,
                        "action": int(action) if hasattr(action, "item") else action,
                        "reward": float(reward),
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                    }
                    episode_replay["steps"].append(step_data)

                obs = new_obs

            # Finalize episode replay data
            if episode_replay is not None:
                episode_replay["total_reward"] = float(episode_reward)
                episode_replay["episode_length"] = episode_length
                replay_data.append(episode_replay)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(
                f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}"
            )

        # Save replay data if enabled
        if (
            hasattr(self.config, "save_replay")
            and self.config.save_replay
            and replay_data
        ):
            replay_filename = f"replay_{self.config.eval_subject or self.config.subject}_{self.config.eval_exercise or self.config.exercise}_episodes.json"
            replay_path = os.path.join(self.config.save_path, replay_filename)

            os.makedirs(os.path.dirname(replay_path), exist_ok=True)

            replay_summary = {
                "metadata": {
                    "subject": self.config.eval_subject or self.config.subject,
                    "exercise": self.config.eval_exercise or self.config.exercise,
                    "model_path": model_path,
                    "n_episodes": n_episodes,
                    "env_config": {
                        "c": self.config.c,
                        "m": self.config.m,
                        "tol": self.config.tol,
                        "dataset_root": self.config.dataset_root,
                    },
                },
                "episodes": replay_data,
            }

            with open(replay_path, "w") as f:
                json.dump(replay_summary, f, indent=2)

            print(f"Replay data saved to: {replay_path}")

        print(f"\nEvaluation Results ({n_episodes} episodes):")
        print(
            f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
        )
        print(
            f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}"
        )

        return episode_rewards, episode_lengths


def load_config_from_yaml(yaml_path):
    """Load configuration from YAML file"""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_config():
    """Create configuration with default hyperparameters"""
    parser = argparse.ArgumentParser(description="Train PPO for rep detection")

    # Configuration file
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Environment parameters
    parser.add_argument("--subject", type=str, help="Subject ID for training")
    parser.add_argument("--exercise", type=str, help="Exercise type for training")
    parser.add_argument(
        "--dataset-root", type=str, default="train", help="Dataset root directory"
    )
    parser.add_argument("--c", type=int, default=10, help="SLS parameter c")
    parser.add_argument("--m", type=int, default=5, help="SLS parameter m")
    parser.add_argument(
        "--tol", type=int, default=10, help="Tolerance for rep detection"
    )

    # Training parameters
    parser.add_argument(
        "--total-timesteps", type=int, default=100000, help="Total training timesteps"
    )
    parser.add_argument(
        "--n-envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--n-steps", type=int, default=2048, help="Steps per environment per update"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--n-epochs", type=int, default=10, help="Number of epochs per update"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value function coefficient"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="Max gradient norm"
    )

    # Model parameters
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--n-layers", type=int, default=2, help="Number of hidden layers"
    )

    # Evaluation parameters
    parser.add_argument("--eval-subject", type=str, help="Subject ID for evaluation")
    parser.add_argument(
        "--eval-exercise", type=str, help="Exercise type for evaluation"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10000, help="Evaluation frequency"
    )
    parser.add_argument(
        "--n-eval-episodes", type=int, default=5, help="Number of evaluation episodes"
    )

    # Saving and logging
    parser.add_argument(
        "--save-path", type=str, default="./models", help="Path to save models"
    )
    parser.add_argument("--load-path", type=str, help="Path to load existing model")
    parser.add_argument(
        "--supervised-path",
        type=str,
        help="Load initial weights from supervised training",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=50000, help="Checkpoint frequency"
    )
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Use TensorBoard logging"
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Mode: train or eval",
    )
    parser.add_argument("--model-path", type=str, help="Path to model for evaluation")
    parser.add_argument(
        "--n-eval-eps", type=int, default=10, help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--save-replay", action="store_true", help="Save replay data during evaluation"
    )
    parser.add_argument(
        "--eval-deterministic",
        action="store_true",
        help="Use deterministic policy during evaluation",
    )

    args = parser.parse_args()

    # Load YAML config if provided
    if args.config:
        yaml_config = load_config_from_yaml(args.config)

        # Override defaults with YAML values (only if not provided via command line)
        for key, value in yaml_config.items():
            key_with_underscores = key.replace("-", "_")
            if hasattr(args, key_with_underscores):
                # Check if the argument was provided via command line by comparing to default
                default_value = parser.get_default(key_with_underscores)
                current_value = getattr(args, key_with_underscores)
                # If current value is same as default, use YAML value
                if current_value == default_value:
                    setattr(args, key_with_underscores, value)

    # Validate required arguments
    if not args.subject:
        raise ValueError("Subject is required (via --subject or config file)")
    if not args.exercise:
        raise ValueError("Exercise is required (via --exercise or config file)")

    return args


def main():
    config = create_config()

    # Create save directory
    os.makedirs(config.save_path, exist_ok=True)

    trainer = PPOTrainer(config)

    if config.mode == "train":
        trainer.train()
    elif config.mode == "eval":
        if not config.model_path:
            raise ValueError("Model path required for evaluation mode")
        trainer.evaluate(config.model_path, config.n_eval_eps)


if __name__ == "__main__":
    main()
