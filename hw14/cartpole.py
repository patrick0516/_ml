import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")  
env.reset(seed=42)

for _ in range(100):
    env.render()
    observation, reward, terminated, truncated, info = env.step(get_action_based_on_observation(env))

    print(f"Observation: {observation}, Reward: {reward}")

    if terminated or truncated:
        print("Episode finished")
        observation, info = env.reset()

env.close()

def get_action_based_on_observation(env):
    pole_angle = env.observation[2]
    
    return 1 if pole_angle > 0 else 0
