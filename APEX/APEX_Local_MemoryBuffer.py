
import numpy as np
import tensorflow as tf

from itertools import chain

class APEX_NStepReturn_MemoryBuffer(object):
    def __init__(self, buffer_size, N, gamma, state_shape, action_shape, action_type = np.float32):
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.next_states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = action_type)
        self.rewards_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.gamma_power_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.td_errors_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.buffer_size = buffer_size
        self.memory_idx = 0
        self.N = N
        self.gammas=[]
        self.gammas.append(1) # gamma**0
        self.gammas.append(gamma) #gamma**1
        for i in range(2, N + 1):
            self.gammas.append(np.power(gamma,i))

    def store(self, state:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        self.states_memory[write_idx] = state
        self.next_states_memory[write_idx] = next_state
        self.actions_memory[write_idx] = action
        self.rewards_memory[write_idx] = 0 # will be set in a loop
        self.gamma_power_memory[write_idx] = 0
        self.dones_memory[write_idx] = is_terminal
        #propogate back current reward
        n_return_idx = 0
        while self.memory_idx - n_return_idx >= 0 and n_return_idx < self.N: # [0 .. N-1]
            self.rewards_memory[self.memory_idx - n_return_idx] += reward * self.gammas[n_return_idx]
            self.gamma_power_memory[self.memory_idx - n_return_idx] = n_return_idx
            n_return_idx += 1
        self.memory_idx += 1

    def update_td_errors(self, idxs, td_errors):
        self.td_errors_memory[idxs] = tf.math.abs(td_errors)

    def reset(self):
        self.memory_idx = 0

    def get_tail_batch(self, transfer_len):
        upper_bound = self.memory_idx % self.buffer_size
        lower_bound = upper_bound - transfer_len
        tail_interval = 0
        if lower_bound < 0:
            tail_interval = np.abs(lower_bound)
            lower_bound = 0
            
        idxs = range(lower_bound, upper_bound)
        if tail_interval > 0:
            idxs = list(chain(idxs, range(self.buffer_size - tail_interval, self.buffer_size)))
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.gamma_power_memory[idxs]), \
            tf.stack(self.dones_memory[idxs]), \
            tf.stack(self.td_errors_memory[idxs])

    def __call__(self, batch_size):
        upper_bound = (self.memory_idx - 1) if self.memory_idx < self.buffer_size else (self.buffer_size - 1)

        #if self.dones_memory[upper_bound] < 1:
        #    upper_bound -= self.N # last N records don't have full n-step return calculated, unless it is a terminal state.
        idxs = np.random.permutation(upper_bound)[:batch_size]
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.gamma_power_memory[idxs]), \
            tf.stack(self.dones_memory[idxs]), \
            idxs