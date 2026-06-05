import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from preprocess import load_data
from environment import EnergyEnv
from qlearning import QLearningAgent

# UI setup
st.set_page_config(page_title="Energy RL", layout="centered")
st.title("⚡ Energy Optimization using Q-Learning")

# Load data
df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Setup
env = EnergyEnv(df)
agent = QLearningAgent()

episodes = st.slider("Training Episodes", 1, 50, 20)

reward_history = []

# Train
if st.button("Train Model"):
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
        st.write(f"Episode {ep+1}: Reward = {round(total_reward,2)}")

    st.success("Training Completed!")

    # Plot
    st.subheader("Reward Trend")

    fig1, ax1 = plt.subplots(figsize=(6, 3))

    if len(reward_history) > 3:
        smoothed = np.convolve(reward_history, np.ones(3)/3, mode='valid')
        ax1.plot(smoothed)
    else:
        ax1.plot(reward_history)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Learning Progress")

    st.pyplot(fig1)

    # Q-table
    st.subheader("Q-Table")
    st.write(agent.q_table)

    # Heatmap
    st.subheader("Q-Table Heatmap")

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.imshow(agent.q_table, cmap="coolwarm")
    ax2.set_title("State-Action Values")
    ax2.set_xlabel("Actions")
    ax2.set_ylabel("States")

    st.pyplot(fig2)