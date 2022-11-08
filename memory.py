import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer(object):

    def __init__(self, capacity, seed=1):
        self.capacity = capacity
        self.position = 0
        self.memory = []
        # Seed for reproducible results
        np.random.seed(seed)

    def push(self, *args):
        # Todo 1
        self.memory.append(Transition(*args))

    def sample_batch(self, batch_size):
        # Todo 2
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_buffer_size(self):
        return len(self.memory)

def main():
    test_buffer = ReplayBuffer(1000, 3)

if __name__=='__main__':
    main()
