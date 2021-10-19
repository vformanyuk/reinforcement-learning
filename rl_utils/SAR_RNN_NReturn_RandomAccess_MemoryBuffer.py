import numpy as np
import tensorflow as tf

class SAR_NStepReturn_RandomAccess_MemoryBuffer(object):
    def __init__(self, distributed_mode:bool, buffer_size:int, N:int, gamma:float, 
                state_shape, action_shape, hidden_state_shape, reward_shape=None, action_type = np.float32,
                trajectory_size=80, burn_in_length=40):
        self._distributed_mode = distributed_mode # when NOT in distributed mode trajectory cache and burn-in memory not cleared after episode end
        self._trajectory_size = trajectory_size
        self._burn_in_len = burn_in_length
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = action_type)
        real_reward_shape = (buffer_size,) if reward_shape == None else (buffer_size, *reward_shape)
        self.rewards_memory = np.empty(shape=real_reward_shape, dtype = np.float32)
        self.gamma_power_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.actor_hidden_states_memory = np.empty(shape=(buffer_size, *hidden_state_shape), dtype = np.float32)
        self.burn_in_memory = []
        self.trajectory_cache = []
        self.memory_idx = 0
        self.hidden_state_idx = 0
        self.N = N
        self.current_trajectory = []
        self.burn_in_trajectory = []
        self.collecting_burn_in = True
        self.gammas=[]
        self.gammas.append(1) # gamma**0
        self.gammas.append(gamma) #gamma**1
        for i in range(2, N + 1):
            self.gammas.append(np.power(gamma,i))

    def store(self, actor_hidden_state:tf.Tensor, state:tf.Tensor, action:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        self.states_memory[self.memory_idx] = state
        self.actions_memory[self.memory_idx] = action
        self.rewards_memory[self.memory_idx] = 0
        self.gamma_power_memory[self.memory_idx] = 0
        self.dones_memory[self.memory_idx] = is_terminal
        #propogate back current reward
        n_return_idx = 0
        while self.memory_idx - n_return_idx >= 0 and n_return_idx < self.N: # [0 .. N-1]
            self.rewards_memory[self.memory_idx - n_return_idx] += reward * self.gammas[n_return_idx]
            self.gamma_power_memory[self.memory_idx - n_return_idx] = n_return_idx
            n_return_idx += 1

        if not self.collecting_burn_in or self._burn_in_len == 0: # Burn-in trajectory must be filled up first if used
            self.current_trajectory.append(self.memory_idx)
        
        if  len(self.current_trajectory) == (self._trajectory_size - self._burn_in_len):
            self.burn_in_trajectory.clear() # clear exisitng burn-in trajectory to start collecting new one
        
        if self._burn_in_len > 0 and len(self.burn_in_trajectory) < self._burn_in_len:
            self.burn_in_trajectory.append(self.memory_idx)
            if len(self.burn_in_trajectory) == 1: # store hidden states for burn-in trajectory unroll
                 self.__store_hidden_state(actor_hidden_state)
            if len(self.burn_in_trajectory) == self._burn_in_len: # save burn-in trajectory and begin collecting training trajectory
                self.__store_burn_in(self.burn_in_trajectory) # don't clear collected trajectory here                
                self.current_trajectory.append(self.memory_idx) # last burn-in trajectory record is first one of training trajectory
                self.collecting_burn_in = False

        if len(self.current_trajectory) == self._trajectory_size or is_terminal > 0: # trajectory shouldn't overlap episode
            if len(self.current_trajectory) == 1: # trajectory must have at least legth of 2
                # if current trajectory length is 1, then burn-in memeory can not contain redundant records
                # thus, it is safe to take preciding state from burn-in to train trajectory
                self.current_trajectory.insert(0, self.memory_idx - 1)
                self.burn_in_memory[-1:].pop()
            self.__cache(self.current_trajectory)
            self.current_trajectory.clear()
            if is_terminal > 0: # this part is redundant for R2D2 agent because it's reset afte each played episode
                redundant_records_count = len(self.burn_in_memory) - len(self.trajectory_cache) if self._burn_in_len > 0 else 0
                assert redundant_records_count >= 0, "To few burn-in trajectories"
                # burn-in and hidden states storages might contain redundant records.
                # Only trajectory store contains correct number of records
                for _ in range(redundant_records_count):
                    self.burn_in_memory.pop()
                    self.hidden_state_idx -= 1
                self.reset()
                return
        self.memory_idx += 1

    def reset(self):
        if self._distributed_mode: # in distribured mode (for APE-X or R2D2) memory completly cleared after every episode
            self.burn_in_memory.clear()
            self.hidden_state_idx = 0
            self.trajectory_cache.clear()
        self.current_trajectory.clear()
        self.burn_in_trajectory.clear()
        self.collecting_burn_in = True
        self.memory_idx = 0

    def __store_burn_in(self, burn_in):
        burn_in_trajectory = []
        for idx in burn_in:
            burn_in_trajectory.append(tf.convert_to_tensor(self.states_memory[idx], dtype=tf.float32))
        self.burn_in_memory.append(burn_in_trajectory)

    def __store_hidden_state(self, hidden_state):
        self.actor_hidden_states_memory[self.hidden_state_idx] = hidden_state
        self.hidden_state_idx+=1

    def __cache(self, trajectory):
        states_idxs = trajectory[:-1]
        assert len(states_idxs) > 0, "Bad trajectory. \"states_idxs\" length=0 when caching trajectory"
        trajectory_idxs = trajectory[1:]
        assert len(trajectory_idxs) > 0, "Bad trajectory. \"trajectory_idxs\" length=0 when caching trajectory"
        states_ = tf.stack(self.states_memory[states_idxs])
        actions_ = tf.stack(self.actions_memory[trajectory_idxs])
        next_states_ = tf.stack(self.states_memory[trajectory_idxs])
        rewards_ = tf.stack(self.rewards_memory[trajectory_idxs])
        gps_ = tf.stack(self.gamma_power_memory[trajectory_idxs])
        dones_ = tf.stack(self.dones_memory[trajectory_idxs])
        self.trajectory_cache.append((states_, actions_, next_states_, rewards_, gps_, dones_))

    def get_all_trajectories(self):
        return self(len(self.trajectory_cache))

    def __call__(self, batch_size):
        idxs = np.random.permutation(len(self.trajectory_cache))[:batch_size]
        for idx in idxs:
            yield self.actor_hidden_states_memory[idx], \
                    self.burn_in_memory[idx], \
                    self.trajectory_cache[idx][0], \
                    self.trajectory_cache[idx][1], \
                    self.trajectory_cache[idx][2], \
                    self.trajectory_cache[idx][3], \
                    self.trajectory_cache[idx][4], \
                    self.trajectory_cache[idx][5]