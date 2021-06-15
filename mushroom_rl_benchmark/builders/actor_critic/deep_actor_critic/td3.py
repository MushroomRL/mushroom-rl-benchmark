import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import TD3
from mushroom_rl.policy import ClippedGaussianPolicy

from mushroom_rl_benchmark.builders import AgentBuilder
from mushroom_rl_benchmark.builders.network import TD3ActorNetwork as ActorNetwork, TD3CriticNetwork as CriticNetwork


class TD3Builder(AgentBuilder):
    """
    AgentBuilder for Twin Delayed DDPG algorithm (TD3)

    """

    def __init__(self, policy_class, policy_params, actor_params, actor_optimizer, critic_params, alg_params,
                 n_steps_per_fit=1., preprocessors=None):
        """
        Constructor.

        Args:
            policy_class (Policy): policy class;
            policy_params (dict): parameters for the policy;
            actor_params (dict): parameters for the actor;
            actor_optimizer (dict): parameters for the actor optimizer;
            critic_params (dict): parameters for the critic;
            alg_params (dict): parameters for the algorithm;
            n_steps_per_fit (int, 1): number of steps per fit.

        """
        self.policy_class = policy_class
        self.policy_params = policy_params
        self.actor_params = actor_params
        self.actor_optimizer = actor_optimizer
        self.critic_params = critic_params
        self.alg_params = alg_params
        super().__init__(n_steps_per_fit=n_steps_per_fit, preprocessors=preprocessors, compute_policy_entropy=False)

    def build(self, mdp_info):
        actor_input_shape = mdp_info.observation_space.shape
        action_scaling = (mdp_info.action_space.high - mdp_info.action_space.low)/2
        self.actor_params['input_shape'] = actor_input_shape
        self.actor_params['output_shape'] = mdp_info.action_space.shape
        self.actor_params['action_scaling'] = action_scaling
        critic_input_shape = (actor_input_shape[0] + mdp_info.action_space.shape[0],)
        self.critic_params["input_shape"] = critic_input_shape
        self.critic_params["action_shape"] = mdp_info.action_space.shape

        if self.policy_class is ClippedGaussianPolicy:
            self.policy_params['sigma'] = np.eye(mdp_info.action_space.shape[0]) * 0.01
            self.policy_params['low'] = mdp_info.action_space.low
            self.policy_params['high'] = mdp_info.action_space.high

        return TD3(mdp_info, self.policy_class, self.policy_params, self.actor_params, self.actor_optimizer,
                   self.critic_params, **self.alg_params)

    def compute_Q(self, agent, states):
        actions = agent._actor_approximator(states)
        q_max = agent._critic_approximator(states, actions)
        return q_max.mean()
    
    @classmethod
    def default(cls, actor_lr=1e-4, actor_network=ActorNetwork, critic_lr=1e-3, critic_network=CriticNetwork,
                initial_replay_size=500, max_replay_size=50000, batch_size=64, n_features=[80, 80], tau=1e-3,
                preprocessors=None, use_cuda=False, get_default_dict=False):
        defaults = locals()
        
        policy_class = ClippedGaussianPolicy
        policy_params = dict()
            
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
            tau=tau)

        builder = cls(policy_class, policy_params, actor_params, actor_optimizer, critic_params, alg_params, preprocessors=preprocessors)

        if get_default_dict:
            return builder, defaults
        else:
            return builder
