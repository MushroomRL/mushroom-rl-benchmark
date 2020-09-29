import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import DDPG
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy

from mushroom_rl_benchmark.agent import AgentBuilder
from mushroom_rl_benchmark.agent.network import DDPGActorNetwork as ActorNetwork, DDPGCriticNetwork as CriticNetwork

class DDPGBuilder(AgentBuilder):
    """
    Builder for Deep Deterministic Policy Gradient algorithm (DDPG).
    """

    def __init__(self, policy_class, policy_params, actor_params, actor_optimizer, critic_params, alg_params, n_steps_per_fit=1): #undefined parameters to default value
        self.policy_class = policy_class
        self.policy_params = policy_params
        self.actor_params = actor_params
        self.actor_optimizer = actor_optimizer
        self.critic_params = critic_params
        self.alg_params = alg_params
        super().__init__(n_steps_per_fit, compute_policy_entropy=False)

    def build(self, mdp_info):
        actor_input_shape = mdp_info.observation_space.shape
        action_scaling = (mdp_info.action_space.high - mdp_info.action_space.low)/2
        self.actor_params['input_shape'] = actor_input_shape
        self.actor_params['output_shape'] = mdp_info.action_space.shape
        self.actor_params['action_scaling'] = action_scaling
        critic_input_shape = (actor_input_shape[0] + mdp_info.action_space.shape[0],)
        self.critic_params["input_shape"] = critic_input_shape
        self.critic_params["action_shape"] = mdp_info.action_space.shape
        return DDPG(mdp_info, self.policy_class, self.policy_params, self.actor_params, self.actor_optimizer, self.critic_params, **self.alg_params)

    def compute_Q(self, agent, states):
        actions = agent._actor_approximator(states)
        q_max = agent._critic_approximator(states, actions)
        return q_max.mean()

    def random_init(self, trial):
        n_features = trial.suggest_categorical('n_features', [32, 64])
        actor_lr = trial.suggest_loguniform('actor_lr', 1e-5, 1e-2)
        critic_lr = trial.suggest_loguniform('critic_lr', 1e-5, 1e-2)

        self.actor_params['n_features'] = n_features
        self.actor_optimizer['params']['lr'] = actor_lr
        self.critic_params['n_features'] = n_features
        self.critic_params['optimizer']['params']['lr'] = critic_lr
    
    @classmethod
    def default(cls, actor_lr=1e-4, actor_network=ActorNetwork, critic_lr=1e-3, critic_network=CriticNetwork, initial_replay_size=500, max_replay_size=50000, batch_size=64, n_features=[80, 80], use_cuda=False):
        
        policy_class = OrnsteinUhlenbeckPolicy
        policy_params = dict(
            sigma=np.ones(1) * .2, 
            theta=.15, dt=1e-2)
            
        actor_params = dict(
            network=actor_network,
            n_features=n_features,
            use_cuda=use_cuda)

        actor_optimizer = {
            'class': optim.Adam,
            'params': {'lr': actor_lr}}

        critic_params = dict(
            network=critic_network,
            optimizer={
                'class': optim.Adam, 
                'params': {'lr': critic_lr}},
            loss=F.mse_loss,
            n_features=n_features,
            output_shape=(1,),
            use_cuda=use_cuda)
        
        alg_params = dict(
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            batch_size=batch_size,
            tau=1e-3)

        return cls(policy_class, policy_params, actor_params, actor_optimizer, critic_params, alg_params)
        