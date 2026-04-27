# buffer.py
# storage class to hold trajectories collected during episode

class rollout_buffer():
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.state_values = []
        self.dones = []