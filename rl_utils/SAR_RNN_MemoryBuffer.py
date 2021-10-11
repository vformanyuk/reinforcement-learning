
import numpy as np
import tensorflow as tf

from itertools import chain

class SAR_NStepReturn_RandomAccess_MemoryBuffer(object):
    def __init__(self, buffer_size, N, gamma, state_shape, action_shape, reward_shape=None, action_type = np.float32, \
                trajectory_size=80, trajectory_overlap=40, burn_in_length=20):
        self._trajectory_size = trajectory_size
        self._trajectory_overlap = trajectory_overlap
        self._burn_in_len = burn_in_length
        self.states_memory = np.empty(shape=(buffer_size, *state_shape), dtype = np.float32)
        self.actions_memory = np.empty(shape=(buffer_size, *action_shape), dtype = action_type)
        real_reward_shape = (buffer_size,) if reward_shape == None else (buffer_size, *reward_shape)
        self.rewards_memory = np.empty(shape=real_reward_shape, dtype = np.float32)
        self.gamma_power_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.dones_memory = np.empty(shape=(buffer_size,), dtype = np.float32)
        self.actor_hidden_states_memory = [] # TODO: replace with dict
        self.buffer_size = buffer_size
        self.memory_idx = 0
        self.N = N
        self.current_trajectory = []
        self.overlapping_trajectory = []
        self.burn_in_trajectory = []
        self.trajectory_store = []
        self.burn_in_store = []
        self.collecting_burn_in = True
        self.hidden_state_idx = 0
        self.gammas=[]
        self.gammas.append(1) # gamma**0
        self.gammas.append(gamma) #gamma**1
        for i in range(2, N + 1):
            self.gammas.append(np.power(gamma,i))

    def store(self, actor_hidden_state:tf.Tensor, state:tf.Tensor, action:tf.Tensor, reward:tf.Tensor, is_terminal:tf.Tensor):
        write_idx = self.memory_idx % self.buffer_size
        self.states_memory[write_idx] = state
        self.actions_memory[write_idx] = action
        self.rewards_memory[write_idx] = 0 # will be set in a loop
        self.gamma_power_memory[write_idx] = 0
        self.dones_memory[write_idx] = is_terminal
        #propogate back current reward
        n_return_idx = 0
        update_idx = (self.buffer_size + (write_idx - n_return_idx)) % self.buffer_size
        while self.memory_idx - n_return_idx >= 0 and n_return_idx < self.N: # [0 .. N-1]
            if self.dones_memory[update_idx] > 0 and update_idx != write_idx: # don't propagate reward to previous episodes, but accept terminal state of current episode
                break
            self.rewards_memory[update_idx] += reward * self.gammas[n_return_idx]
            self.gamma_power_memory[update_idx] = n_return_idx
            n_return_idx += 1
            update_idx = (self.buffer_size + (write_idx - n_return_idx)) % self.buffer_size

        if not self.collecting_burn_in or self._burn_in_len == 0: # Burn-in trajectory must be filled up first if used
            self.current_trajectory.append(write_idx)

        if len(self.current_trajectory) == self._trajectory_size or is_terminal > 0.5: # trajectory shouldn't overlap episode
            # zero length trajectories are problem
            self.trajectory_store.append(np.array(self.current_trajectory[:], dtype=np.int)) # trajectory length is less then expected
            self.current_trajectory = self.overlapping_trajectory[:]
            self.overlapping_trajectory = []

        if len(self.current_trajectory) >= (self._trajectory_size - self._trajectory_overlap): # begin collecting overlaping trajectory
            self.overlapping_trajectory.append(write_idx)
        
        if  len(self.current_trajectory) == (self._trajectory_size - self._burn_in_len  - self._trajectory_overlap):
            self.burn_in_trajectory.clear() # clear exisitng burn-in trajectory to start collecting new one
        
        if self._burn_in_len > 0 and len(self.burn_in_trajectory) < self._burn_in_len:
            self.burn_in_trajectory.append(write_idx)
            if len(self.burn_in_trajectory) == 1: # store hidden states for burn-in trajectory unroll
                 self.actor_hidden_states_memory.append(actor_hidden_state)
            if len(self.burn_in_trajectory) == self._burn_in_len: # save burn-in trajectory and begin collecting training trajectory
                self.burn_in_store.append(np.array(self.burn_in_trajectory[:], dtype=np.int))
                self.current_trajectory.append(write_idx) # last burn-in trajectory record is first one of training trajectory
                self.collecting_burn_in = False

        if is_terminal > 0:
            redundant_records_count = len(self.burn_in_store) - len(self.trajectory_store)
            # burn-in and hidden states storages might contain redundant records.
            # Only trajectory store contains correct number of records
            for _ in range(redundant_records_count):
                self.burn_in_store.pop() 
                self.actor_hidden_states_memory.pop()
            self.current_trajectory.clear()
            self.overlapping_trajectory.clear()
            self.burn_in_trajectory.clear()
            self.collecting_burn_in = True

        self.memory_idx += 1

    def __call__(self, batch_size):
        # TODO: burn-in trajectories are less then training trajectories. Fix required.
        idxs = np.random.permutation(len(self.trajectory_store))[:batch_size]
        for idx in idxs:
            # first state in trajectory is taken from burn-in period
            states_idxs = self.trajectory_store[idx][:-1]
            trajectory_idxs = self.trajectory_store[idx][1:]
            yield self.actor_hidden_states_memory[idx], \
                    self.states_memory[self.burn_in_store[idx]], \
                    tf.stack(self.states_memory[states_idxs]), \
                    tf.stack(self.actions_memory[trajectory_idxs]), \
                    tf.stack(self.states_memory[trajectory_idxs]), \
                    tf.stack(self.rewards_memory[trajectory_idxs]), \
                    tf.stack(self.gamma_power_memory[trajectory_idxs]), \
                    tf.stack(self.dones_memory[trajectory_idxs])