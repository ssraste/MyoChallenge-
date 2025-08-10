import os
import gym
import numpy as np
import json
from myosuite.utils import gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt  

env = gym.make('myoChallengeRelocateP1-v0')

# to stabilize simulation
class ActionClipWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)

# Custom reward function 
def compute_custom_reward(env, info):
    distance_to_target = info.get('distance', 0)
    object_grasped = info.get('grasp_success', False)
    placed_at_target = info.get('placed_at_target', False)
    time_penalty = -0.01
    
    hand_position = info.get('hand_position', np.zeros(3))
    object_position = info.get('object_position', np.zeros(3))

    hand_object_distance = np.linalg.norm(hand_position - object_position)
    hand_object_reward = -hand_object_distance * 5
    distance_reward = -distance_to_target * 10
    grasp_bonus = 20 if object_grasped else 0
    place_bonus = 50 if placed_at_target else 0

    intermediate_reward = 0
    if distance_to_target < 0.5:
        intermediate_reward += 5

    reward = hand_object_reward + distance_reward + grasp_bonus + intermediate_reward + time_penalty + place_bonus

    return reward

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        custom_reward = compute_custom_reward(self.env, info)
        return obs, custom_reward, terminated, truncated, info

env = ActionClipWrapper(env)
env = CustomRewardWrapper(env)

model_filename = "ppo_myo_grasping.zip"
metrics_filename = "ppo_training_metrics.json"
metrics = []

evaluation_metrics_filename = "evaluation_metrics.json"
evaluation_metrics = []

if os.path.exists(metrics_filename):
    with open(metrics_filename, "r") as f:
        try:
            metrics = json.load(f)
        except json.JSONDecodeError:
            metrics = []

if os.path.exists(model_filename):
    print(f"Loading existing model from '{model_filename}'...")
    model = PPO.load(model_filename, env=env)
else:
    print("No saved model found. Creating a new model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0004,
        n_steps=2048,
        batch_size=128,
        clip_range=0.2,
        ent_coef=0.01,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        target_kl=0.02
    )

def evaluate_model(env, model, num_episodes=5):
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)
            if isinstance(result, tuple):
                obs, reward, done, info = result[:4]
            else:
                obs, reward, done, info = result[0], result[1], result[2], result[3]
            episode_reward += reward
            if isinstance(obs, tuple):
                obs = obs[0]
        episode_rewards.append(episode_reward)
    
    avg_reward = np.mean(episode_rewards)
    return avg_reward

# Real-time plot setup
plt.ion()  # Enable interactive mode
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Initialize empty lists for plotting
timesteps = []
value_loss = []
policy_loss = []
avg_rewards = []

print("Training the PPO model...")
total_timesteps = 50000
timesteps_per_render = 100
current_timestep = 0

while current_timestep < total_timesteps:
    model.learn(total_timesteps=timesteps_per_render, reset_num_timesteps=False)
    ep_len_mean = model.logger.name_to_value.get("rollout/ep_len_mean", None)
    ep_rew_mean = model.logger.name_to_value.get("rollout/ep_rew_mean", None)

    if ep_len_mean is None or ep_rew_mean is None:
        ep_len_mean = env.get_attr('episode_length', 0) if hasattr(env, 'get_attr') else None
        ep_rew_mean = env.get_attr('episode_rewards', 0) if hasattr(env, 'get_attr') else None

    metric_entry = {
        "timesteps": current_timestep,
        "ep_len_mean": ep_len_mean,
        "ep_rew_mean": ep_rew_mean,
        "value_loss": model.logger.name_to_value.get("train/value_loss", None),
        "policy_loss": model.logger.name_to_value.get("train/policy_gradient_loss", None)
    }
    metrics.append(metric_entry)

    with open(metrics_filename, "a") as f:  
        f.write(json.dumps(metric_entry) + "\n")  
        f.flush()  
    print(f"Metrics saved up to timestep {current_timestep}.")

    # Evaluate the model every 1000 timesteps and for 5 episodes
    if current_timestep % 1000 == 0:
        avg_reward = evaluate_model(env, model, num_episodes=5) 
        print(f"Evaluation at timestep {current_timestep}: Average reward: {avg_reward}")
        
        evaluation_metrics.append({
            "timesteps": current_timestep,
            "avg_reward": avg_reward
        })
        
        with open(evaluation_metrics_filename, "a") as eval_file:
            eval_file.write(json.dumps({
                "timesteps": current_timestep,
                "avg_reward": avg_reward
            }) + "\n")

    current_timestep += timesteps_per_render

    timesteps.append(current_timestep)
    value_loss.append(metric_entry["value_loss"])
    policy_loss.append(metric_entry["policy_loss"])

    avg_reward = evaluate_model(env, model, num_episodes=5)
    avg_rewards.append(avg_reward)
    
    ax1.clear()
    ax1.plot(timesteps, value_loss, label="Value Loss", color='red')
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Value Loss")
    ax1.set_title("Value Loss vs Timesteps")
    ax1.legend()

    ax2.clear()
    ax2.plot(timesteps, policy_loss, label="Policy Loss", color='blue')
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Policy Loss")
    ax2.set_title("Policy Loss vs Timesteps")
    ax2.legend()

    ax3.clear()
    ax3.plot(timesteps, avg_rewards, label="Average Reward", color='green')
    ax3.set_xlabel("Timesteps")
    ax3.set_ylabel("Average Reward")
    ax3.set_title("Average Reward vs Timesteps")
    ax3.legend()

    plt.pause(0.1)

    with open(metrics_filename, "a") as f:
        f.write(json.dumps(metric_entry) + "\n")
        f.flush()

    evaluation_metrics.append({
        "timesteps": current_timestep,
        "avg_reward": avg_reward
    })
    with open(evaluation_metrics_filename, "a") as eval_file:
        eval_file.write(json.dumps({
            "timesteps": current_timestep,
            "avg_reward": avg_reward
        }) + "\n")

    current_timestep += timesteps_per_render

model.save(model_filename)
print(f"Model saved as '{model_filename}'.")

plt.ioff()
plt.show()
