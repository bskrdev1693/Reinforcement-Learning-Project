
## Reinforcement Learning Locomotion Experiments

### Overview

The project is a comparison between reinforcement learning (RL) agents in simulated continuous control locomotion tasks. The experiments are performed on the Gymnasium MuJoCo environments HalfCheetah-v4 and Hopper-v4 using Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC).

The program trains multiple agents using multiple random seeds, tests their policies, writes metrics to CSV files, and produces plots to compare reward, learning progress, sample efficiency, and agent training time.

### Features

Performs comparisons between PPO and SAC in two MuJoCo environments.
For all configurations, run them with random seeds 11 and 22.
Checks policies every 1000 training timesteps.
Periodically evaluates with 10 episodes, and tests with 50 episodes.
Normalizes observations, using VecNormalize.
Saves trained models, normalization statistics, evaluation records and final test results.
Plots reward and learning-curve plots, as well as sample-efficiency and training-time plots.
Contains a custom residual feature extractor (SimbaFeatureExtractor) for an optional SAC-Simba configuration.

### Requirements

The project requires Python and compatible versions of the following libraries:

NumPy
pandas
Matplotlib
Gymnasium
Stable-Baselines3
PyTorch
MuJoCo

### Evaluation Metrics

A few metrics are used to compare the performance of reinforcement learning in the project.

Final Mean Reward: Average return over the last 50 Episode Evaluation episodes. The higher the value, the better the performance.

Reward Standard Deviation: Takes care of the range of returns to final episodes. A lower variation may be a sign of more consistent performance.

Median Reward: This is the median (50th percentile) of the return of the episodes from the results of the last evaluation.

Mean Episode Length: Average number of steps recorded for each evaluation episode of the end of the test.

Episode Completion Rate: Percentage that does not include the number of episodes that are terminated by the environment.

Learning-Curve AUC: Average area of the periodic mean-reward curve, normalized by the number of the timesteps the learning has been configured for.

Training Time: The actual time spent training each agent (wall clock).

All of the metrics enable comparison on metrics of final performance, learning progress, consistency, sample efficiency, and computational cost.


### Reproducibility

The program resets random seeds for the python, numpy, pytorch and stable-baselines3 libraries. For each of the default experimental configurations two seeds are used.

First, there's a small number of seeds for comparison. More seeds and more training runs can give better evidence of the differences in performance on the different agents.

### Conclusion

The project aims to set up a framework that allows for a comparison of the performance of PPO and SAC in simulated environments of robotic locomotion. The training results, evaluation metrics, CSV summaries, and visualizations aid in analyzing the performance of reward, learning efficiency, and computational needs.