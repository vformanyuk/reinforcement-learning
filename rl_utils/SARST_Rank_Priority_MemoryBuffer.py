import numpy as np
import tensorflow as tf
import bisect as bs
from functools import total_ordering

@total_ordering
class __rank_container:
    def __init__(self, replay_buffer_idx, abs_td_error, probability=0., *args, **kwargs):
        self.replay_buffer_idx = replay_buffer_idx
        self.td_error = abs_td_error
        self.probability = probability
    def __lt__(self, value):
        return self.td_error < value.td_error
    def __eq__(self, value):
        #return math.isclose(self.td_error, value.td_error, rel_tol=1e-6)
        return self.replay_buffer_idx == value.replay_buffer_idx

class SARST_Rank_Priority_MemoryBuffer(object):
    def __init__(self, buffer_size, state_shape, action_shape, action_type = np.float32, alpha=0.7, beta=0.5, beta_increase_rate=1.0001):
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.next_states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = action_type)
        self.rewards_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.buffer_size = buffer_size
        self.memory_idx = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_inc_rate = beta_increase_rate
        self.td_max = 1
        self.max_is_weight = 1
        self.total_rank = 0.
        self.ordered_storage = list()

    def store(self, state:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        self.states_memory[write_idx] = state
        self.next_states_memory[write_idx] = next_state
        self.actions_memory[write_idx] = action
        self.rewards_memory[write_idx] = reward
        self.dones_memory[write_idx] = is_terminal
        self.total_rank += 1 #P(i)=1/(rank(i)=0+1)=>1^alpha
        if self.memory_idx >= self.buffer_size:
            pass #TODO get item with replay_buffer_idx==write_idx, decrease total_rank by it's rank^alpha, remove item from sorted_container
        self.ordered_storage.insert(0, __rank_container(write_idx, self.td_max, 1./self.total_rank)) # p(i)=(1/rank(i))^alpha / sum(P^alpha)
        self.memory_idx += 1

    def update_priorities(self, idxs, td_errors):
        # get replay buffer index
        # delete from bisected array by index
        # insert new td error with index from replay buffer got on step 1
        for idx, err in zip(idxs, np.abs(td_erros)):
            self.total_rank -= np.power(1/idx, self.alpha)
            if err > self.td_max:
                self.td_max = err
            update_idx = self.ordered_storage.pop(idx).replay_buffer_idx
            container = __rank_container(update_idx, err)
            rank = bs.bisect(self.ordered_storage, container)
            priority = np.power(1/rank, self.alpha)
            self.total_rank += priority
            container.probability = priority/self.total_rank
            self.ordered_storage.insert(rank,container)

    def __call__(self, batch_size):
        segment_len = len(self.ordered_storage) / batch_size #TODO incorrect sampling. Segments should be equal by probability, not by length
        idxs = list()
        importance_sampling_weights = list()
        random_priorities = [segment_len * (j + r) for j,r in enumerate(np.random.uniform(low=0, high=1, size=batch_size))]

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
