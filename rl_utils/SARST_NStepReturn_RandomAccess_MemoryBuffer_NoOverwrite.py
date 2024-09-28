
import numpy as np
import tensorflow as tf

from itertools import chain

class SARST_NStepReturn_RandomAccess_MemoryBuffer_NoOverwrite(object):
    def __init__(self, buffer_size, N, gamma, state_shape, action_shape, action_type = np.float32):
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.next_states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = action_type)
        self.rewards_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.gamma_power_memory = np.empty(shape=(buffer_size,), dtype = np.int32)
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
        self.actions_memory[write_idx] = action
        self.next_states_memory[write_idx] = next_state
        self.rewards_memory[write_idx] = reward
        self.gamma_power_memory[write_idx] = 0
        self.dones_memory[write_idx] = is_terminal
        gamma_idx = 1
        write_idx -= 1
        n_counter = self.N
        done = is_terminal < 1
        while write_idx >=0 and n_counter > 0:
            if self.dones_memory[write_idx] > 0 and done: # don't propagate reward to previous episodes, but accept terminal state of current episode
                break
            self.rewards_memory[write_idx] += reward * self.gammas[gamma_idx]
            self.gamma_power_memory[write_idx] = gamma_idx
            self.next_states_memory[write_idx] = next_state
            self.dones_memory[write_idx] = is_terminal
            gamma_idx += 1
            write_idx -= 1
            n_counter -= 1
        if done:
            fix_idx = self.memory_idx % self.buffer_size
            for n in range(1, self.N):
                self.rewards_memory[fix_idx] += sum([reward * self.gammas[g] for g in range(n, self.N)])
                self.gamma_power_memory[fix_idx] = self.N - 1
                fix_idx -= 1
        self.memory_idx += 1

    def reset(self):
        self.memory_idx = 0

    def get_tail_batch(self, batch_size):
        upper_bound = self.memory_idx % self.buffer_size
        lower_bound = max(upper_bound - batch_size, 0)
        idxs = range(lower_bound, upper_bound)
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.gamma_power_memory[idxs]), \
            tf.stack(self.dones_memory[idxs])

    def __len__(self):
        return self.memory_idx + 1

    def __call__(self, batch_size):
        upper_bound = (self.memory_idx - 1) if self.memory_idx < self.buffer_size else (self.buffer_size - 1)
        if self.dones_memory[upper_bound] < 1:
            upper_bound -= self.N # last N records don't have full n-step return calculated, unless it is a terminal state.
        idxs = np.random.permutation(upper_bound)[:batch_size]
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.gamma_power_memory[idxs]), \
            tf.stack(self.dones_memory[idxs])