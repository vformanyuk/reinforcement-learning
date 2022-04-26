import numpy as np
import tensorflow as tf

class SARST_RandomAccess_MemoryBuffer(object):
    def __init__(self, buffer_size, state_shape, action_shape, action_type = np.float32):
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.next_states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = action_type)
        self.rewards_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.buffer_size = buffer_size
        self.memory_idx = 0

    def store(self, state:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        self.states_memory[write_idx] = state
        self.next_states_memory[write_idx] = next_state
        self.actions_memory[write_idx] = action
        self.rewards_memory[write_idx] = reward
        self.dones_memory[write_idx] = is_terminal
        self.memory_idx += 1

    def __call__(self, batch_size, idxs = []):
        if len(idxs) == 0:
            upper_bound = self.memory_idx if self.memory_idx < self.buffer_size else self.buffer_size
            idxs = np.random.permutation(upper_bound)[:batch_size]
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.dones_memory[idxs])

class SARST_MultiTask_RandomAccess_MemoryBuffer(SARST_RandomAccess_MemoryBuffer):
    def __init__(self, buffer_size, state_shape, action_shape, context_vector_length, action_type = np.float32):
        super().__init__(buffer_size,state_shape, action_shape, action_type)
        self.tasks_memory = np.empty(shape=(buffer_size, context_vector_length), dtype = np.float32)
    def store(self, context_vector:tf.Tensor, state:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        self.tasks_memory[write_idx] = context_vector
        super().store(state,action,next_state,reward,is_terminal)
    def __call__(self, batch_size):
        upper_bound = self.memory_idx if self.memory_idx < self.buffer_size else self.buffer_size
        idxs = np.random.permutation(upper_bound)[:batch_size]
        context_vectors = tf.stack(self.tasks_memory[idxs])
        states, actions, next_states, rewards, dones = super().__call__(batch_size, idxs)
        return states, actions, next_states, rewards, dones, context_vectors