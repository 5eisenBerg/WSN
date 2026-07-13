import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        """Initializes the experience replay buffer memory."""
        self.buffer = deque(maxlen=capacity)
 
    def push(self, s, a, r, ns, d):
        """Stores a standard transition tuple into memory."""
        self.buffer.append((
            np.array(s, dtype=np.float32).flatten(), 
            int(a), 
            float(r), 
            np.array(ns, dtype=np.float32).flatten(), 
            float(d)
        ))
 
    def sample(self, batch_size):
        """Randomly samples a batch of transitions for neural network training."""
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.stack(s), np.array(a), np.array(r), np.stack(ns), np.array(d)

    def __len__(self):
        """Returns the current size of internal memory storage."""
        return len(self.buffer)
