class EnergyEnv:
    def __init__(self, data):
        # Use only required column (avoid string issues)
        self.data = data["Global_active_power"].values
        self.index = 0

    def reset(self):
        self.index = 0
        return self.get_state()

    def get_state(self):
        power = float(self.data[self.index])

        if power < 1:
            return 0  # low
        elif power < 3:
            return 1  # medium
        else:
            return 2  # high

    def step(self, action):
        power = float(self.data[self.index])

        # Normalize
        power = power / 10

        # Base reward
        reward = -power

        state = self.get_state()

        # Smart reward system
        if state == 2 and action == 3:
            reward += 5
        elif state == 1 and action == 2:
            reward += 3
        elif state == 0 and action == 1:
            reward += 1
        else:
            reward -= 1

        self.index += 1
        done = self.index >= len(self.data)

        next_state = self.get_state() if not done else 0

        return next_state, reward, done