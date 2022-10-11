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
        self.gamma_powers_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory_idx = 0
        self.alpha_target = alpha
        self.current_alpha = 0
        self.beta = beta
        self.beta_inc_rate = beta_increase_rate
        self.max_is_weight = 1
        self.ordered_storage = list()
        self.lookup = dict()
        self.internal_ordering_counter = 0
        self.internal_ordering_threshold = 500

    '''
    Calculate analogue of Euler–Mascheroni constant for chosen alpha
    '''
    def __get_gamma_s(self, alpha):
        pm_s = 1.01956 + 0.223632*alpha + 3.45985 * 1e-2 *np.power(alpha,2) - 9.32331*1e-4*np.power(alpha,2) - 1.40047*1e-5*np.power(alpha,3) + 7.63*1e-6*np.power(alpha,4)
        return 2*np.arctan(pm_s)/np.pi # Calculate analogue of Euler–Mascheroni constant for chosen alpha

    '''
    Paper on N-th harmonic number approximation
    http://mi.mathnet.ru/eng/irj251
    '''
    def __get_partial_sum_aproximation(self, buffer_size, batch_size, alpha, gamma_s):
        aprox_total = (np.power(buffer_size,1 - alpha) - 1)/(1-alpha) + gamma_s # approximation of N-th harmonic number (partial sum of generalized harmonic series)
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
    __get_sampling_interval_boundary(0, a, l=segment_len)=72
    __get_sampling_interval_boundary(1, a, l=segment_len)=267
    ...
    __get_sampling_interval_boundary(K, a, l=segment_len)=999999
    '''
    def __get_sampling_interval_boundary(self, interval_idx, interval_len, gamma_s, alpha):
        return np.exp(np.log((interval_len*interval_idx - gamma_s)*(1-alpha) + 1) / (1-alpha))

    def store(self, state:tf.Tensor, action:tf.Tensor, next_state:tf.Tensor, reward:tf.Tensor, gamma_power:tf.Tensor, is_terminal:tf.Tensor, td_error:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        self.states_memory[write_idx] = state
        self.next_states_memory[write_idx] = next_state
        self.actions_memory[write_idx] = action
        self.rewards_memory[write_idx] = reward
        self.gamma_powers_memory[write_idx] = gamma_power
        self.dones_memory[write_idx] = is_terminal
        if self.memory_idx >= self.buffer_size:
            self.lookup[write_idx].td_error = td_error
            self.internal_ordering_counter += 1

            if self.internal_ordering_counter > self.internal_ordering_threshold:
                self.ordered_storage.sort()
                self.internal_ordering_counter = 0
        else:
            container = rank_container(write_idx, td_error)
            bs.insort_right(self.ordered_storage, container) # O(logN) + O(n)
            self.lookup[write_idx] = container
        assert len(self.ordered_storage) <= self.buffer_size, f"[Store] Mem index = {self.memory_idx}, Buff size = {self.buffer_size}"
        self.memory_idx += 1

    def update_priorities(self, meta_idxs, td_errors):
        to_remove = []
        # because indexes are fetched in reversed way, items are poped from the end and thus array indexes of preciding items are not affected
        for idx in meta_idxs:
            to_remove.append(self.ordered_storage.pop(idx))
        for container, err in zip(to_remove, td_errors):
            bs.insort_right(self.ordered_storage, rank_container(container.replay_buffer_idx, err)) # O(logN) + O(n)
        assert len(self.ordered_storage) <= self.buffer_size, f"[Update] Mem index = {self.memory_idx}, Buff size = {self.buffer_size}"

    def __len__(self):
        return self.memory_idx

    def __call__(self):
        idxs = list()
        meta_idxs = list()
        importance_sampling_weights = np.empty(shape=(self.batch_size,), dtype = np.float32)

        # aprox_total used for IS weights calculation, segment length - for segment boundaries
        storage_len = len(self.ordered_storage)
        
        # alpha_coef = 1
        # alpha = alpha_coef * (storage_len / self.buffer_size)
       
        alpha_sigma_coef1 = 5.0
        alpha_sigma_coef2 = (self.buffer_size / 2.0) / (2*alpha_sigma_coef1)
        alpha = 1 - tf.math.sigmoid(alpha_sigma_coef1 - storage_len / alpha_sigma_coef2).numpy()

        alpha = min(alpha, self.alpha_target)
        self.current_alpha = alpha

        gamma_s = self.__get_gamma_s(alpha)
        aprox_total, segment_len = self.__get_partial_sum_aproximation(storage_len, self.batch_size, alpha, gamma_s)
        assert aprox_total > 0

        interval_start = 0
        for k in range(1, self.batch_size + 1):
            boundary = int(np.ceil(self.__get_sampling_interval_boundary(k, segment_len, gamma_s, alpha)))
            interval_low = max(storage_len - boundary, 0)
            interval_high = storage_len - interval_start

            assert interval_low < interval_high, f'{interval_low} => {interval_high}, boundary = {boundary}, interval_start = {interval_start} loop step - {k}, alpha - {alpha}, segment_len = {segment_len}, storage_len = {storage_len}'

            meta_idx = np.random.randint(low=interval_low, high=interval_high, size=1)[0] #reverse intervals because higher error records are in the end
            container = self.ordered_storage[meta_idx]
            idxs.append(container.replay_buffer_idx)
            is_weight = np.power(self.buffer_size * np.power(meta_idx + 1, -alpha) / aprox_total, -self.beta) # meta_idx could be 0, so use 'meta_idx+1' for calculations instead
            if is_weight > self.max_is_weight:
                self.max_is_weight = is_weight
            importance_sampling_weights[k - 1] = is_weight
            meta_idxs.append(meta_idx)
            interval_start = (interval_start + 1) if boundary <= interval_start else boundary
        
        self.beta = self.beta * self.beta_inc_rate
        if self.beta > 1:
            self.beta = 1
        
        #normalize IS weights
        return tf.stack(self.states_memory[idxs]), \
            tf.stack(self.actions_memory[idxs]), \
            tf.stack(self.next_states_memory[idxs]), \
            tf.stack(self.rewards_memory[idxs]), \
            tf.stack(self.gamma_powers_memory[idxs]), \
            tf.stack(self.dones_memory[idxs]), \
            tf.convert_to_tensor(importance_sampling_weights / self.max_is_weight, dtype=tf.float32), \
            meta_idxs
