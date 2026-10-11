"""Experimentally tested RL agents in simulated environments, comparing rewards, training time, learning progress, and learning efficiency.
The results of the performance were stored in CSV files and the graphs were plotted to compare the performance of the agents in each of the experiments."""

# Importing the required libraries
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
import glob
import torch
import torch.nn as nn
import stable_baselines3

import stable_baselines3
import gymnasium
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import sync_envs_normalization


# Setting reproducible experiment parameters
ENVIRONMENTS = ["Hopper-v4", "HalfCheetah-v4"]
AGENTS = ["PPO", "SAC"]
SEEDS = [11, 22]

TOTAL_STEPS = 10000
EVAL_FREQUENCY = 1000
EVAL_EPISODES = 10
FINAL_EPISODES = 50

RESULTS_DIR = "rl_locomotion_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Creating a normalized residual feature extractor
class SimbaFeatureExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space,
        features_dim=128
    ):
        super().__init__(observation_space, features_dim)

        input_dim = int(np.prod(observation_space.shape))
        hidden_dim = 128

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim)
        )

    def forward(self, observations):

        x = self.input_norm(
            self.input_layer(observations)
        )

        x = x + self.block1(x)
        x = x + self.block2(x)

        return self.output_layer(x)


# Creating a monitored MuJoCo environment
def make_environment(env_name, seed):

    def initialize():

        env = gym.make(env_name)
        env = Monitor(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)

        return env

    return initialize


# Evaluating a trained policy over multiple episodes
def evaluate_policy(model, eval_env, episodes=10):

    episode_returns = []
    episode_lengths = []
    episode_terminations = []

    for _ in range(episodes):

        obs = eval_env.reset()
        done = np.array([False])
        total_reward = 0.0
        length = 0
        terminated_early = False

        while not done[0]:

            action, _ = model.predict(
                obs, deterministic=True
            )

            obs, rewards, dones, infos = eval_env.step(action)

            total_reward += float(rewards[0])
            length += 1

            if dones[0]:
                terminated_early = bool(
                    infos[0].get("terminated", False)
                )
                done = dones

        episode_returns.append(total_reward)
        episode_lengths.append(length)
        episode_terminations.append(terminated_early)

    return (
        np.asarray(episode_returns),
        np.asarray(episode_lengths),
        np.asarray(episode_terminations)
    )


# Recording periodic evaluation results
class EvaluationCallback(BaseCallback):

    def __init__(
        self,
        eval_env,
        eval_frequency,
        eval_episodes,
        log_path
    ):
        super().__init__()

        self.eval_env = eval_env
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.log_path = log_path
        self.records = []

    def _on_step(self):

        if self.num_timesteps % self.eval_frequency < self.training_env.num_envs:

            sync_envs_normalization(
                self.training_env, self.eval_env
            )

            returns, lengths, terminations = evaluate_policy(
                self.model,
                self.eval_env,
                self.eval_episodes
            )

            record = {
                "timesteps": self.num_timesteps,
                "mean_reward": np.mean(returns),
                "std_reward": np.std(returns, ddof=1),
                "median_reward": np.median(returns),
                "min_reward": np.min(returns),
                "max_reward": np.max(returns),
                "mean_episode_length": np.mean(lengths),
                "episode_completion_rate": (
                    1.0 - np.mean(terminations)
                )
            }

            self.records.append(record)

            print(
                f"Steps: {self.num_timesteps:,} | "
                f"Mean reward: {record['mean_reward']:.2f}"
            )

            pd.DataFrame(self.records).to_csv(
                self.log_path, index=False
            )

        return True


# Creating and training the selected agent
def train_agent(env_name, agent_name, seed):

    print("\n" + "=" * 65)
    print(
        f"Environment: {env_name} | "
        f"Agent: {agent_name} | Seed: {seed}"
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    run_name = f"{env_name}_{agent_name}_seed{seed}"
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Creating separate training and evaluation environments
    train_env = DummyVecEnv([
        make_environment(env_name, seed)
    ])

    eval_env = DummyVecEnv([
        make_environment(env_name, seed + 10000)
    ])

    # Normalizing observations consistently across agents
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )

    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )

    # Configuring the selected reinforcement learning agent
    if agent_name == "PPO":

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            seed=seed,
            verbose=0,
            device=DEVICE
        )

    elif agent_name == "SAC":

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            seed=seed,
            verbose=0,
            device=DEVICE
        )

    elif agent_name == "SAC-Simba":

        policy_kwargs = {
            "features_extractor_class": SimbaFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": 128
            },
            "net_arch": {
                "pi": [128, 128],
                "qf": [128, 128]
            }
        }

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=0,
            device=DEVICE
        )

    else:
        raise ValueError("Unknown agent")

    # Measuring training time
    callback = EvaluationCallback(
        eval_env=eval_env,
        eval_frequency=EVAL_FREQUENCY,
        eval_episodes=EVAL_EPISODES,
        log_path=os.path.join(run_dir, "evaluation.csv")
    )

    start_time = time.time()

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=callback
    )

    training_seconds = time.time() - start_time

    # Evaluating the final policy on fresh episodes
    sync_envs_normalization(train_env, eval_env)

    final_returns, final_lengths, final_terminations = (
        evaluate_policy(
            model,
            eval_env,
            FINAL_EPISODES
        )
    )

    # Saving the trained policy and normalization statistics
    model.save(os.path.join(run_dir, "model"))
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))

    pd.DataFrame({
        "episode": np.arange(1, FINAL_EPISODES + 1),
        "return": final_returns,
        "length": final_lengths,
        "terminated_early": final_terminations
    }).to_csv(
        os.path.join(run_dir, "final_test.csv"),
        index=False
    )

    # Calculating the final performance measures
    evaluation_df = pd.DataFrame(callback.records)

    if len(evaluation_df) >= 2:

        # Calculating normalized learning-curve area
        auc = np.trapz(
            evaluation_df["mean_reward"].to_numpy(),
            evaluation_df["timesteps"].to_numpy()
        ) / TOTAL_STEPS

    else:
        auc = np.nan

    summary = {
        "environment": env_name,
        "agent": agent_name,
        "seed": seed,
        "training_steps": TOTAL_STEPS,
        "final_mean_reward": np.mean(final_returns),
        "final_std_reward": np.std(final_returns, ddof=1),
        "final_median_reward": np.median(final_returns),
        "learning_curve_auc": auc,
        "mean_episode_length": np.mean(final_lengths),
        "episode_completion_rate": (
            1.0 - np.mean(final_terminations)
        ),
        "training_seconds": training_seconds
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(run_dir, "summary.csv"),
        index=False
    )

    train_env.close()
    eval_env.close()

    print("Final mean reward:", summary["final_mean_reward"])
    print("Training time:", round(training_seconds, 2), "seconds")

    return summary


# Running all experimental configurations
if __name__ == "__main__":

    all_results = []

    for env_name in ENVIRONMENTS:
        for agent_name in AGENTS:
            for seed in SEEDS:

                result = train_agent(
                    env_name,
                    agent_name,
                    seed
                )

                all_results.append(result)

                pd.DataFrame(all_results).to_csv(
                    os.path.join(
                        RESULTS_DIR,
                        "all_experiment_results.csv"
                    ),
                    index=False
                )

    print("\nAll training experiments completed.")
    print(pd.DataFrame(all_results))



RESULTS_DIR = "rl_locomotion_results"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Loading the combined experiment results
results_file = os.path.join(
    RESULTS_DIR, "all_experiment_results.csv"
)

results = pd.read_csv(results_file)

print("Experiment Summary")
print(results.head())

print("\nSummary Statistics")

summary = results.groupby(
    ["environment", "agent"]
).agg(
    mean_reward=("final_mean_reward", "mean"),
    std_reward=("final_mean_reward", "std"),
    mean_auc=("learning_curve_auc", "mean"),
    mean_training_time=("training_seconds", "mean")
).reset_index()

print(summary.round(3))

summary.to_csv(
    os.path.join(PLOT_DIR, "performance_summary.csv"),
    index=False
)

# Plot 1: Comparing final rewards
for env_name in results["environment"].unique():

    subset = results[results["environment"] == env_name]

    agent_stats = subset.groupby("agent")[
        "final_mean_reward"
    ].agg(["mean", "std"])

    plt.figure(figsize=(9, 5))

    plt.bar(
        agent_stats.index,
        agent_stats["mean"],
        yerr=agent_stats["std"].fillna(0),
        capsize=5
    )

    plt.title(f"Final Mean Reward: {env_name}")
    plt.xlabel("RL Agent")
    plt.ylabel("Mean Episode Reward")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR, f"{env_name}_final_rewards.png"
        ),
        dpi=300
    )
    plt.show()

# Plot 2: Learning curves across random seeds
evaluation_files = glob.glob(
    os.path.join(RESULTS_DIR, "*", "evaluation.csv")
)

curve_data = []

for file in evaluation_files:

    run_name = os.path.basename(os.path.dirname(file))

    agent = next(
        (name for name in ["PPO", "SAC"]
         if name in run_name),
        None
    )

    environment = next(
        (name for name in ["Hopper-v4", "HalfCheetah-v4"]
         if name in run_name),
        None
    )

    if agent is None or environment is None:
        continue

    try:
        seed = int(run_name.split("seed")[-1])
    except ValueError:
        seed = -1

    df = pd.read_csv(file)

    if not {"timesteps", "mean_reward"}.issubset(df.columns):
        print("Skipping file with missing columns:", file)
        continue

    df["agent"] = agent
    df["environment"] = environment
    df["seed"] = seed

    curve_data.append(df)

if curve_data:

    curves = pd.concat(curve_data, ignore_index=True)

    for env_name in curves["environment"].unique():

        plt.figure(figsize=(10, 6))

        env_data = curves[
            curves["environment"] == env_name
        ]

        for agent in env_data["agent"].unique():

            agent_data = env_data[
                env_data["agent"] == agent
            ]

            grouped = agent_data.groupby("timesteps")[
                "mean_reward"
            ]

            mean_curve = grouped.mean()
            std_curve = grouped.std().fillna(0)

            plt.plot(
                mean_curve.index,
                mean_curve.values,
                marker="o",
                markersize=3,
                label=agent
            )

            plt.fill_between(
                mean_curve.index,
                mean_curve.values - std_curve.values,
                mean_curve.values + std_curve.values,
                alpha=0.15
            )

        plt.title(f"Learning Curves: {env_name}")
        plt.xlabel("Training Timesteps")
        plt.ylabel("Evaluation Reward")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOT_DIR, f"{env_name}_learning_curves.png"
            ),
            dpi=300
        )
        plt.show()

# Plot 3: Sample efficiency comparison
if curve_data:

    auc_records = []

    for (environment, agent, seed), group in curves.groupby(
        ["environment", "agent", "seed"]
    ):
        group = group.sort_values("timesteps")

        if len(group) >= 2:
            normalized_auc = np.trapz(
                group["mean_reward"].to_numpy(),
                group["timesteps"].to_numpy()
            ) / max(group["timesteps"].max(), 1)
        else:
            normalized_auc = np.nan

        auc_records.append({
            "environment": environment,
            "agent": agent,
            "seed": seed,
            "normalized_auc": normalized_auc
        })

    auc_summary = pd.DataFrame(auc_records)

    auc_summary.to_csv(
        os.path.join(PLOT_DIR, "sample_efficiency_summary.csv"),
        index=False
    )

    for env_name in auc_summary["environment"].unique():

        subset = auc_summary[
            auc_summary["environment"] == env_name
        ]

        plt.figure(figsize=(9, 5))

        for agent in subset["agent"].unique():

            values = subset[
                subset["agent"] == agent
            ]["normalized_auc"].dropna()

            plt.scatter(
                [agent] * len(values),
                values,
                label=agent
            )

        plt.title(f"Sample Efficiency: {env_name}")
        plt.xlabel("RL Agent")
        plt.ylabel("Normalized Learning-Curve AUC")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(PLOT_DIR, f"{env_name}_auc.png"),
            dpi=300
        )

        plt.show()

else:
    print("No evaluation.csv files found. Learning curves were skipped.")

# Plot 4: Training time comparison
for env_name in results["environment"].unique():

    subset = results[results["environment"] == env_name]

    times = subset.groupby("agent")[
        "training_seconds"
    ].agg(["mean", "std"])

    plt.figure(figsize=(9, 5))

    plt.bar(
        times.index,
        times["mean"],
        yerr=times["std"].fillna(0),
        capsize=5
    )

    plt.title(f"Training Time: {env_name}")
    plt.xlabel("RL Agent")
    plt.ylabel("Training Time (seconds)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOT_DIR, f"{env_name}_training_time.png"),
        dpi=300
    )
    plt.show()

print("\nAnalysis completed.")
print("Plots and summary files saved in:", PLOT_DIR)