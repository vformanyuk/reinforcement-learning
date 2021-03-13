# reinforcement-learning

Some reinforcement learning algorithms implemented with Tensorflow 2

For practise I chose [OpenAI gym](https://github.com/openai/gym) Lunar Lander environment.
It is representative and doesn't skip frames like some other envs.

* [Double DQN](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_doubleDQN.py)
* [Double Dueling DQN](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_double_dueling_DQN.py)
* [Double Dueling DQN with priority based importance sampling](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_double_dueling_DQN_IS.py)
* [Double Dueling DQN with rank based importance sampling](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_double_dueling_DQN_IS_rank.py)
* [Vanila Policy Gradient](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_PolicyGradient.py)
* [Advantage Actor-Critic with MC update](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_ActorCritic.py)
* [Advantage Actor-Critic with online update, N-returns backup and entropy bonus](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_a2c_tdn_entropy.py)
* [Advantage Actor-Critic with online update, N-returns backup implemented in replay buffer and entropy bonus](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_a2c_tdn_buffer_with_entropy.py)
* [Proximal Policy Optimization(PPO) + Generalized Advantage Estimator(GAE)](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_PPO.py)
* [Deep Deterministic Policy Gradient (DDPG)](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_DDPG.py)
* [Twin Delayed Deep Deterministic policy gradient (TD3)](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_TD3.py)
* [Soft Actor-Critic (SAC)](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_SAC.py)
* APE-X DPG
  * [Orchestrator](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_ape-x.py)
  * [DPG Actor](https://github.com/vformanyuk/reinforcement-learning/blob/master/APEX/dpg_actor_slim.py)
  * [DPG Learner](https://github.com/vformanyuk/reinforcement-learning/blob/master/APEX/dpg_learner.py)
* APE-X with Soft Actor Critic
  * [Orchestrator](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_ape-x-SAC.py)
  * [SAC Actor](https://github.com/vformanyuk/reinforcement-learning/blob/master/APEX/sac_actor.py)
  * [SAC Learner](https://github.com/vformanyuk/reinforcement-learning/blob/master/APEX/sac_learner.py)
* [Curiosity based on Random Network Distillation ( with Soft Actor Critic)](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_RND_Curiosity.py)
