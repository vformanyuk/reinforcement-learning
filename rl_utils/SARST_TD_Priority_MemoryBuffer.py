import numpy as np
import tensorflow as tf

class SARST_TD_Priority_MemoryBuffer(object):
    def __init__(self, buffer_size, state_shape, action_shape, action_type = np.float32, alpha=0.6, beta=0.4, beta_increase_rate=1.0001, greedy_sampling = False):
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.next_states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = action_type)
        self.rewards_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.buffer_size = buffer_size #must be a power of 2
        self.memory_idx = 0
        self.epsilon = 1e-6
        self.alpha = alpha
        self.beta = beta
        self.beta_inc_rate = beta_increase_rate
        self.__sampling_method = self.__stohastic_sample
        if greedy_sampling:
            self.__sampling_method = self.__greedy_sample
        self.tree_capacity = buffer_size
        self._sum_tree = [0]*(self.tree_capacity * 2 - 1)
        for i in range(self.tree_capacity):
            self._sum_tree[i] = (0,0)
        self.td_max = 1
        self.max_is_weight = 1

    def store(self, state:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        self.states_memory[write_idx] = state
        self.next_states_memory[write_idx] = next_state
        self.actions_memory[write_idx] = action
        self.rewards_memory[write_idx] = reward
        self.dones_memory[write_idx] = is_terminal
        self.memory_idx += 1
        self._propagate(write_idx, self.td_max - self._sum_tree[write_idx][0])
        self._sum_tree[write_idx] = (self.td_max, write_idx)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs,td_errors):
            priority = np.power(np.abs(err) + self.epsilon, self.alpha)
            if priority > self.td_max:
                self.td_max = priority
            self._propagate(idx, priority - self._sum_tree[idx][0])
            self._sum_tree[idx] = (priority, idx)

    def _propagate(self, idx, change):
        if idx >= len(self._sum_tree) - 1:
            return
        parentIdx = self.tree_capacity + idx // 2
        self._sum_tree[parentIdx] += change
        self._propagate(parentIdx, change)

    @property
    def _total(self):
        return self._sum_tree[len(self._sum_tree) - 1]

    def _get(self, search_num):
        search = float(search_num)
        lvl_base_idx = len(self._sum_tree) - 1
        lvl_idx = 0
        search_idx = lvl_base_idx
        lvl=1
        while lvl_base_idx>=self.tree_capacity:
            lvl_base_idx -= 1<<lvl
            left_child_idx = lvl_base_idx + lvl_idx * 2
            right_child_idx = left_child_idx + 1
            
            left_child = self._sum_tree[left_child_idx]
            if left_child_idx < self.tree_capacity:
                left_child = self._sum_tree[left_child_idx][0]

            if left_child > search:
                search_idx = left_child_idx
            else:
                search_idx = right_child_idx
                search -= left_child
            if lvl_base_idx > 0:
                lvl_idx = search_idx % lvl_base_idx
            lvl+=1
        return self._sum_tree[search_idx]
    def __greedy_sample(self, batch_size):
        return np.random.uniform(low=self.epsilon, high=self._total, size=batch_size)
    def __stohastic_sample(self, batch_size):
        segment_len = self._total / batch_size
        return list([segment_len * (j + r) for j,r in enumerate(np.random.uniform(low=0, high=1, size=batch_size))])
    def __call__(self, batch_size):
        random_priorities = self.__sampling_method(batch_size)
        idxs = list()
        importance_sampling_weights = list()

        for p in random_priorities:
            data_item = self._get(p)
            idxs.append(data_item[1])
            is_weight = np.power(self.buffer_size * (data_item[0]/self._total), -self.beta)
            if is_weight > self.max_is_weight:
                self.max_is_weight = is_weight
            importance_sampling_weights.append(is_weight / self.max_is_weight)
        
        self.beta = self.beta * self.beta_inc_rate
        if self.beta > 1:
            self.beta = 1
        
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.dones_memory[idxs]), \
            tf.convert_to_tensor(importance_sampling_weights, dtype=tf.float32), \
            idxs
