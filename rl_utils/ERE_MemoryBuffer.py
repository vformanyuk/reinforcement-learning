'''
https://arxiv.org/abs/1906.04009
'''

import numpy as np
import tensorflow as tf

class ERE_MemoryBuffer(object):
    def __init__(self, buffer_size, state_shape, action_shape, action_type = np.float32, n = 0.996):
        self.states_memory = np.empty(shape=(2, buffer_size, *state_shape), dtype = np.float32)
        self.next_states_memory = np.empty(shape=(2, buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(2, buffer_size, *action_shape), dtype = action_type)
        self.rewards_memory = np.empty(shape=(2, buffer_size), dtype = np.float32)
        self.dones_memory = np.empty(shape=(2, buffer_size), dtype = np.float32)

        self.ere_n = n
        self.buffer_size = buffer_size
        self.memory_idx = 0

    def store(self, state:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        array_idx = (self.memory_idx // self.buffer_size) % 2
        self.states_memory[array_idx, write_idx] = state
        self.next_states_memory[array_idx, write_idx] = next_state
        self.actions_memory[array_idx, write_idx] = action
        self.rewards_memory[array_idx, write_idx] = reward
        self.dones_memory[array_idx, write_idx] = is_terminal
        self.memory_idx += 1

    def __call__(self, batch_size, K):
        insertion_round = self.memory_idx // self.buffer_size
        array_idx = insertion_round % 2
        write_idx = self.memory_idx % self.buffer_size
        upper_bound = self.memory_idx
        if insertion_round > 0:
            upper_bound = self.buffer_size

        for k in range(1, K + 1):
            ck = max(int(upper_bound * np.power(self.ere_n, (k/K) * 1000)), 2 * batch_size)
            inner_idxs = np.random.permutation(ck)[:batch_size]
            idxs = [(array_idx, write_idx - i) if write_idx - i >= 0 else (array_idx - 1, self.buffer_size + (write_idx - i)) for i in inner_idxs]
        
            yield tf.stack([self.states_memory[i] for i in idxs]), \
                tf.stack([self.actions_memory[i] for i in idxs]), \
                tf.stack([self.next_states_memory[i] for i in idxs]), \
                tf.stack([self.rewards_memory[i] for i in idxs]), \
                tf.stack([self.dones_memory[i] for i in idxs])