import gym
import numpy as np
from dqn_agent import DQNAgent

#CartPole-v1, MountainCar-v0, LunarLander-v2
env = gym.make('MountainCar-v0',render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen (gunakan model terlatih jika tersedia)
agent = DQNAgent(state_size, action_size)
agent.epsilon = 0.01  # Minimalkan eksplorasi saat testing

for e in range(50):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])

    for time in range(500):
        env.render()
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)  # ✅ BENAR untuk Gym terbaru
        done = terminated or truncated  # Gabungkan status selesai
        state = np.reshape(next_state, [1, state_size])

        if done:

            print(f"Test Episode: {e+1}, Score: {time}")

            break

env.close()