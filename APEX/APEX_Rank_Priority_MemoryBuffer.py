import numpy as np
import tensorflow as tf
import bisect as bs
from functools import total_ordering

@total_ordering
class rank_container(object):
    def __init__(self, replay_buffer_idx:int, abs_td_error:float, *args, **kwargs):
        self.replay_buffer_idx = int(replay_buffer_idx)
        self.td_error = float(abs_td_error)
    def __lt__(self, value):
        return self.td_error < value.td_error
    def __eq__(self, value):
        return self.replay_buffer_idx == value.replay_buffer_idx

class APEX_Rank_Priority_MemoryBuffer(object):
    def __init__(self, buffer_size, batch_size, state_shape, action_shape, action_type = np.float32, alpha=0.7, beta=0.5, beta_increase_rate=1.0001):
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.next_states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = action_type)
        self.rewards_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory_idx = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_inc_rate = beta_increase_rate
        self.gamma_s = self.__get_gamma_s()
        self.max_is_weight = 1
        self.ordered_storage = list()
        self.lookup = dict()

    '''
    Calculate analogue of Euler–Mascheroni constant for chosen alpha
    '''
    def __get_gamma_s(self):
        pm_s = 1.01956 + 0.223632*self.alpha + 3.45985 * 1e-2 *np.power(self.alpha,2) - 9.32331*1e-4*np.power(self.alpha,2) \
               - 1.40047*1e-5*np.power(self.alpha,3) + 7.63*1e-6*np.power(self.alpha,4)
        return 2*np.arctan(pm_s)/np.pi # Calculate analogue of Euler–Mascheroni constant for chosen alpha

    '''
    Paper on N-th harmonic number approximation
    http://mi.mathnet.ru/eng/irj251
    '''
    def __get_partial_sum_aproximation(self, buffer_size, batch_size, alpha):
        aprox_total = (np.power(buffer_size,1 - alpha) - 1)/(1-alpha) + self.gamma_s # approximation of N-th harmonic number (partial sum of generalized harmonic series)
        segment_len = aprox_total / batch_size
        return aprox_total, segment_len

    #def __calculate_sampling_intervals(self, buffer_size, batch_size, alpha):
    #    aprox_total, segment_len = self.__get_partial_sum_aproximation(buffer_size, batch_size, alpha)
    #    sampling_intervals = list()

    #    prev_boundary = 0
    #    for k in range(batch_size-1):
    #        boundary = int(np.ceil(self.__get_sampling_interval_boundary(k, segment_len)))
    #        shifted_boundary = (prev_boundary+1) if boundary <= prev_boundary else boundary
    #        sampling_intervals.append(shifted_boundary)
    #        prev_boundary = shifted_boundary
    #    sampling_intervals.append(buffer_size - 1)

    #    return sampling_intervals

    '''
    Calculate N-th harmonic series element, sum of which and all before is equal to l*k
    This guarantees that starting sampling interval will be at least of length "l"

    When given a segment length, it also calculates ending interval index for a given k.
    Example:
    N=10**6
    K=128
    segment_len = aprox_total / batch_size
    __get_threshold(0, a, l=segment_len)=72
    __get_threshold(1, a, l=segment_len)=267
    ...
    __get_threshold(K, a, l=segment_len)=999999
    '''
    def __get_sampling_interval_boundary(self, interval_idx, interval_len):
        return np.exp(np.log((interval_len*interval_idx - self.gamma_s)*(1-self.alpha) + 1) / (1-self.alpha))

    def store(self, state:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor, td_error:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        self.states_memory[write_idx] = state
        self.next_states_memory[write_idx] = next_state
        self.actions_memory[write_idx] = action
        self.rewards_memory[write_idx] = reward
        self.dones_memory[write_idx] = is_terminal
        if self.memory_idx >= self.buffer_size:
            self.ordered_storage.remove(self.lookup[write_idx]) # O(n)
        container = rank_container(write_idx, td_error)
        bs.insort_right(self.ordered_storage, container) # O(n)
        self.lookup[write_idx] = container
        self.memory_idx += 1

    def update_priorities(self, meta_idxs, td_errors):
        for idx, err in zip(meta_idxs, np.abs(td_errors)):
            container_to_remove = self.ordered_storage.pop(idx)
            bs.insort_right(self.ordered_storage, rank_container(container_to_remove.replay_buffer_idx, err)) # O(n)

    def __call__(self):
        idxs = list()
        meta_idxs = list()
        importance_sampling_weights = np.empty(shape=(self.batch_size,), dtype = np.float32)

        # aprox_total used for IS weights calculation, segment length - for segment boundaries
        aprox_total, segment_len = self.__get_partial_sum_aproximation(len(self.ordered_storage), self.batch_size, self.alpha)
        storage_len = len(self.ordered_storage)
        interval_start = 0
        for k in range(self.batch_size):
            boundary = int(np.ceil(self.__get_sampling_interval_boundary(k, segment_len)))
            interval_end = (interval_start+1) if boundary <= interval_start else boundary
            meta_idx = np.random.randint(low=storage_len - interval_end, high=storage_len - interval_start, size=1)[0] #reverse intervals because higher error records are in the end
            container = self.ordered_storage[meta_idx]
            idxs.append(container.replay_buffer_idx)
            is_weight = np.power(self.buffer_size * np.power(meta_idx,-self.alpha) / aprox_total, -self.beta)
            if is_weight > self.max_is_weight:
                self.max_is_weight = is_weight
            importance_sampling_weights[k] = is_weight
            meta_idxs.append(meta_idx)
            interval_start = interval_end
        
        self.beta = self.beta * self.beta_inc_rate
        if self.beta > 1:
            self.beta = 1
        
        #normalize IS weights
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.dones_memory[idxs]), \
            tf.convert_to_tensor(importance_sampling_weights / self.max_is_weight, dtype=tf.float32), \
            meta_idxs
