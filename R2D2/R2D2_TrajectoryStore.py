import numpy as np
from numpy.lib.function_base import append
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

class R2D2_TrajectoryStore(object):
    def __init__(self, buffer_size:int, alpha=0.5, beta=0.7, beta_increase_rate=1.000001):
        self.buffer_size = buffer_size
        self.actor_hidden_states_memory = []
        self.burn_in_memory = []
        self.trajectory_cache = []
        self.trajectory_length_memory = []
        self.memory_idx = 0
        # Importance Sampling
        self.alpha = alpha
        self.beta = beta
        self.beta_inc_rate = beta_increase_rate
        self.gamma_s = self.__get_gamma_s()
        self.td_max = 1
        self.max_is_weight = 1
        self.ordered_storage = list()
        self.lookup = dict()
        self.store_counter = 0
        self.store_threshold = 500

    def __len__(self):
        return len(self.trajectory_cache)

    def store(self, actor_hidden_state, burn_in_trajectory, trajectory, trajectory_length, td_error):
        write_idx = self.memory_idx % self.buffer_size
        self.actor_hidden_states_memory.append(actor_hidden_state)
        self.burn_in_memory.append(burn_in_trajectory)
        self.trajectory_cache.append(trajectory)
        self.trajectory_length_memory.append(trajectory_length)
        if self.memory_idx >= self.buffer_size:
            self.lookup[write_idx].td_error = td_error
            self.store_counter += 1
            if self.store_counter >= self.store_threshold:
                self.ordered_storage.sort()
                self.store_counter = 0
        else:
            container = rank_container(write_idx, td_error)
            bs.insort_right(self.ordered_storage, container)
            self.lookup[write_idx] = container
        self.memory_idx += 1

    def get_trajectories_count(self):
        return len(self.trajectory_cache)

    def __get_gamma_s(self):
        pm_s = 1.01956 + 0.223632*self.alpha + 3.45985 * 1e-2 *(self.alpha**2) - 9.32331*1e-4*(self.alpha**2) - 1.40047*1e-5*(self.alpha**3) +7.63*1e-6*(self.alpha**4)
        return 2*np.arctan(pm_s)/np.pi # Calculate analogue of Euler–Mascheroni constant for chosen alpha
    
    def __get_partial_sum_aproximation(self, buffer_size, batch_size, alpha):
        aprox_total = (np.power(buffer_size,1 - alpha) - 1)/(1-alpha) + self.gamma_s # approximation of N-th harmonic number (partial sum of generalized harmonic series)
        segment_len = aprox_total / batch_size
        return aprox_total, segment_len

    def __get_sampling_interval_boundary(self, interval_idx, interval_len):
        return np.exp(np.log((interval_len*interval_idx - self.gamma_s)*(1-self.alpha) + 1) / (1-self.alpha))

    def update_priorities(self, meta_idxs, td_errors):
        to_remove = []
        # because indexes are fetched in reversed way, items are poped from the end and thus array indexes of preciding items are not affected
        for idx in meta_idxs:
            to_remove.append(self.ordered_storage.pop(idx))
        for container, err in zip(to_remove, np.abs(td_errors)):
            if err > self.td_max:
                self.td_max = err
            bs.insort_right(self.ordered_storage, rank_container(container.replay_buffer_idx, err))

    def sample(self, batch_size):
        idxs = list()
        meta_idxs = list()
        importance_sampling_weights = np.empty(shape=(batch_size,), dtype = np.float32)

        # aprox_total used for IS weights calculation, segment length - for segment boundaries
        aprox_total, segment_len = self.__get_partial_sum_aproximation(len(self.ordered_storage), batch_size, self.alpha)
        storage_len = len(self.ordered_storage)
        interval_start = 0
        for k in range(batch_size):
            boundary = int(np.ceil(self.__get_sampling_interval_boundary(k, segment_len)))
            interval_end = (interval_start+1) if boundary <= interval_start else boundary
            meta_idx = np.random.randint(low=max(storage_len - interval_end, 0), high=storage_len - interval_start, size=1)[0] #reverse intervals because higher error records are in the end
            container = self.ordered_storage[meta_idx]
            idxs.append(container.replay_buffer_idx)
            is_weight = np.power(self.buffer_size * np.power(meta_idx,-self.alpha) / aprox_total, -self.beta)
            if is_weight > self.max_is_weight:
                self.max_is_weight = is_weight
            importance_sampling_weights[k] = is_weight
            meta_idxs.append(meta_idx)
            interval_start = interval_end
        
        self.beta = min(1, self.beta * self.beta_inc_rate)
        IS_weights_IT = iter(importance_sampling_weights)
        for idx in idxs:
            normalized_IS_weight = float(next(IS_weights_IT) / self.max_is_weight)
            yield self.actor_hidden_states_memory[idx], \
                    self.burn_in_memory[idx], \
                    self.trajectory_cache[idx], \
                    tf.fill(dims=[self.trajectory_length_memory[idx]], value=normalized_IS_weight), \
                    idx