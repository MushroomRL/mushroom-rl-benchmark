import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import A2C
from mushroom_rl.policy import GaussianTorchPolicy

from mushroom_rl_benchmark.builders import AgentBuilder
from mushroom_rl_benchmark.builders.network import A2CNetwork as Network


class A2CBuilder(AgentBuilder):
    """
    AgentBuilder for Advantage Actor Critic algorithm (A2C)

    """

    def __init__(self, policy_params, actor_optimizer, critic_params, alg_params, n_steps_per_fit=5,
                 preprocessors=None):
        """
        Constructor.

        Args:
            policy_params (dict): parameters for the policy;
            actor_optimizer (dict): parameters for the actor optimizer;
            critic_params (dict): parameters for the critic;
            alg_params (dict): parameters for the algorithm;
            n_steps_per_fit (int, 5): number of steps per fit;
            preprocessors (list, None): list of preprocessors.

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
        return A2C(mdp_info, policy, **self.alg_params)

    def compute_Q(self, agent, states):
        return agent._V(states).mean()
    
    @classmethod
    def default(cls, actor_lr=7e-4, critic_lr=7e-4, critic_network=Network, n_features=64,
                preprocessors=None, use_cuda=False, get_default_dict=False):
        defaults = locals()
        
        policy_params = dict(
            std_0=1.,
            n_features=n_features,
            use_cuda=False)

        actor_optimizer = {
            'class': optim.RMSprop,
            'params': {'lr': actor_lr, 'eps': 3e-3}}

        critic_params = dict(
            network=critic_network,
            optimizer={
                'class': optim.RMSprop, 
                'params': {'lr': critic_lr, 'eps': 1e-5}},
            loss=F.mse_loss,
            n_features=n_features,
            batch_size=64,
            output_shape=(1,))
        
        alg_params = dict(
            max_grad_norm=0.5,
            ent_coeff=1e-2)

        builder = cls(policy_params, actor_optimizer, critic_params, alg_params, preprocessors=preprocessors)

        if get_default_dict:
            return builder, defaults
        else:
            return builder
