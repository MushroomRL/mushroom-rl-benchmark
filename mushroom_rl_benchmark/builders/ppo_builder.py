import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import PPO
from mushroom_rl.policy import GaussianTorchPolicy

from mushroom_rl_benchmark.builders import AgentBuilder
from mushroom_rl_benchmark.builders.network import TRPONetwork as Network


class PPOBuilder(AgentBuilder):
    """
    AgentBuilder for Proximal Policy Optimization algorithm (PPO).
    """

    def __init__(self, policy_params, actor_optimizer, critic_params, alg_params, n_steps_per_fit=3000,
                 preprocessors=None):
        """
        Constructor.

        Args:
            policy_params (dict): parameters for the policy
            actor_optimizer (dict): parameters for the actor optimizer
            critic_params (dict): parameters for the critic
            alg_params (dict): parameters for the algorithm

        Kwargs:
            n_steps_per_fit (int): number of steps per fit (Default: 3000)
            preprocessors (list): list of preprocessors (Default: [StandardizationPreprocessor])

        Returns:
            ppo_builder: AgentBuilder for PPO
        """
        self.policy_params = policy_params
        self.actor_optimizer = actor_optimizer
        self.critic_params = critic_params
        self.alg_params = alg_params
        super().__init__(n_steps_per_fit, preprocessors=preprocessors)

    def build(self, mdp_info):
        policy = GaussianTorchPolicy(
            Network,
            mdp_info.observation_space.shape,
            mdp_info.action_space.shape,
            **self.policy_params)
        self.critic_params["input_shape"] = mdp_info.observation_space.shape
        self.alg_params['critic_params'] = self.critic_params
        self.alg_params['actor_optimizer'] = self.actor_optimizer
        return PPO(mdp_info, policy, **self.alg_params)

    def compute_Q(self, agent, states):
        return agent._V(states).mean()
    
    @classmethod
    def default(cls, actor_lr=3e-4, critic_lr=3e-4, critic_fit_params=None, critic_network=Network, lam=.95,
                n_features=32, n_steps_per_fit=3000, preprocessors=None, use_cuda=False):
        
        policy_params = dict(
            std_0=1.,
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
            batch_size=64,
            output_shape=(1,))

        alg_params = dict(
            n_epochs_policy=4,
            batch_size=64,
            eps_ppo=.2,
            lam=lam,
            critic_fit_params=critic_fit_params)

        return cls(policy_params, actor_optimizer, critic_params, alg_params,
                   n_steps_per_fit=n_steps_per_fit, preprocessors=preprocessors)
