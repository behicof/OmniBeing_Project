
import numpy as np
import random

class ReinforcementLearningCore:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        self.q_table = {}  # کلید: state, مقدار: لیست Q برای هر action
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]
        return self.q_table[state]

    def select_action(self, state):
        q_values = self.get_q_values(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        action_index = self.actions.index(action)
        q_values[action_index] += self.alpha * (reward + self.gamma * max(next_q_values) - q_values[action_index])
