import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size=3, action_size=4):
        self.state_size = state_size
        self.action_size = action_size

        self.q_table = np.zeros((state_size, action_size))

        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state, action] += self.lr * (target - predict)

        # Epsilon decay
        if self.epsilon > 0.01:
            self.epsilon *= 0.995