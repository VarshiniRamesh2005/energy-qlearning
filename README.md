# Energy Optimization using Q-Learning

A reinforcement learning project that uses Q-Learning to optimize household energy consumption. This project trains an intelligent agent to make optimal decisions about energy usage based on real-world power consumption data.

## Project Overview

This application implements a **Q-Learning agent** trained on household power consumption data to learn optimal energy management strategies. The agent learns to take appropriate actions (1-4) based on current power consumption states (low, medium, high) to minimize energy waste.

## Features

- **Q-Learning Algorithm**: Implements a reinforcement learning agent with epsilon-greedy exploration
- **Interactive Streamlit UI**: User-friendly interface for training and visualization
- **Real Data**: Uses household power consumption dataset for realistic training
- **Visualization**: 
  - Reward trend plots with smoothing
  - Q-Table heatmaps for analyzing learned policies
  - Training progress monitoring

## Project Structure

```
.
├── app.py                              # Streamlit web application
├── environment.py                      # EnergyEnv simulation environment
├── qlearning.py                        # QLearningAgent implementation
├── preprocess.py                       # Data loading and preprocessing
├── requirements.txt                    # Python dependencies
├── household_power_consumption.txt     # Dataset
└── README.md                           # This file
```

## Components

### `app.py` - Streamlit Application
Main interactive interface that:
- Loads and previews the dataset
- Initializes the environment and agent
- Trains the Q-Learning agent for specified episodes
- Visualizes training progress and learned Q-Table

### `environment.py` - Energy Environment
Simulates the household energy consumption environment:
- **States**: 3 states based on power consumption levels
  - State 0: Low power (< 1 kW)
  - State 1: Medium power (1-3 kW)
  - State 2: High power (> 3 kW)
- **Actions**: 4 possible actions for energy management
- **Rewards**: Smart reward system that incentivizes appropriate actions for each state

### `qlearning.py` - Q-Learning Agent
Implements the Q-Learning algorithm:
- **State Size**: 3 (low, medium, high power states)
- **Action Size**: 4
- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.9
- **Exploration Rate (ε)**: Decays over time from 1.0 to 0.01

### `preprocess.py` - Data Preprocessing
Handles loading and preprocessing of household power consumption data.

## Installation

1. Clone or download this repository:
```bash
cd reinforcement_learning
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- streamlit
- pandas
- numpy
- matplotlib

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser. You can then:

1. View the dataset preview
2. Adjust the number of training episodes (1-50)
3. Click "Train Model" to start training
4. Monitor reward progression for each episode
5. Analyze the learned Q-Table and visualizations

## How Q-Learning Works

The Q-Learning agent learns by:

1. **Exploration**: Initially takes random actions to explore the environment
2. **Exploitation**: Gradually favors actions with high Q-values
3. **Learning**: Updates Q-values based on observed rewards using:
   ```
   Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
   ```

The agent improves over episodes by learning which actions lead to better long-term rewards in each state.

## Training Results

The reward trend plot shows the agent's learning progress:
- Initial episodes may have variable rewards as the agent explores
- Over time, the smoothed reward curve should show improvement
- The Q-Table heatmap visualizes the learned state-action values

## Future Enhancements

- Deep Q-Learning (DQN) implementation
- Multi-agent scenarios
- Real-time energy management
- Integration with actual smart home systems
- Comparison with other RL algorithms
