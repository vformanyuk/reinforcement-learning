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
* [Soft Actor-Critic with Value network (alpha term regularization taken from SAC v2)](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_SAC_value.py)
* [Soft Actor-Critic with Emphasizing Recent Experience](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_SAC_ERE.py)
* APE-X DPG
  * [Orchestrator](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_ape-x.py)
  * [DPG Actor](https://github.com/vformanyuk/reinforcement-learning/blob/master/APEX/dpg_actor_slim.py)
  * [DPG Learner](https://github.com/vformanyuk/reinforcement-learning/blob/master/APEX/dpg_learner.py)
* APE-X with Soft Actor Critic
  * [Orchestrator](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_ape-x-SAC.py)
  * [SAC Actor](https://github.com/vformanyuk/reinforcement-learning/blob/master/APEX/sac_actor.py)
  * [SAC Learner](https://github.com/vformanyuk/reinforcement-learning/blob/master/APEX/sac_learner.py)
* [Curiosity based on Random Network Distillation ( with Soft Actor Critic)](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_RND_Curiosity.py)
* Recurrent Experience Replay in Distributed Reinforcement Lerning (R2D2) with SAC.
  * [Orchestrator](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_SAC_R2D2.py)
  * [Agent](https://github.com/vformanyuk/reinforcement-learning/blob/master/R2D2/R2D2_SAC_Agent.py)
  * [Learner](https://github.com/vformanyuk/reinforcement-learning/blob/master/R2D2/R2D2_SAC_Learner.py)
  * [Agent buffer](https://github.com/vformanyuk/reinforcement-learning/blob/master/R2D2/R2D2_AgentBuffer.py) Responsible for collecting trajectories and is important part of whole algorithm.
  
  Note 1: for this experiment the famous Lunar Lander environment was altered to produce 'stacked' states. This achived by adding liner interpolated states between 'state' and 'next_state'.

  Note 2: Because original paper says nothing about behavior near trajoctory end, the simplest approach was taken - length of last trajectoy may vary, but has length of atleast 2 records.
* [Regularizing Action Policies for Smooth Control implementation based on Soft Actor-Critic](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_SAC_CAPS.py)
* Active dendrites networks implementation [arxiv paper](https://arxiv.org/abs/2201.00042)
  * Modified LunarLander environemnt [LunarLander multitask](https://github.com/vformanyuk/reinforcement-learning/blob/master/env/lunar_lander.py). This implementation has two tasks: original landing task and new - lift off. Last one require lander to fly off from landing pad
  * [Active Dendrits and k-Winner-Takes-All layers](https://github.com/vformanyuk/reinforcement-learning/blob/master/rl_utils/ad_layer.py)
  * The training [script](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_ActiveDendritsNetwork.py)
* [Quantile Regression QR-DQN](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_QR_DQN.py)
* [Implicit Quantile Networks IQN](https://github.com/vformanyuk/reinforcement-learning/blob/master/lunar_lander_IQN.py)
