import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import TRPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

from mushroom_rl_benchmark.builders import AgentBuilder
from mushroom_rl_benchmark.builders.network import TRPONetwork as Network


class TRPOBuilder(AgentBuilder):
    """
    Builder for Trust Region Policy optimization algorithm (TRPO).
    """

    def __init__(self, policy_params, critic_params, alg_params, n_steps_per_fit=3000, preprocessors=[StandardizationPreprocessor]):
        self.policy_params = policy_params
        self.critic_params = critic_params
        self.alg_params = alg_params
        super().__init__(n_steps_per_fit, preprocessors=preprocessors)

    def build(self, mdp_info):
        #print(self.policy_params)
        policy = GaussianTorchPolicy(
            Network,
            mdp_info.observation_space.shape,
            mdp_info.action_space.shape,
            **self.policy_params)
        self.critic_params["input_shape"] = mdp_info.observation_space.shape
        self.alg_params['critic_params'] = self.critic_params
        #print(self.alg_params)
        return TRPO(mdp_info, policy, **self.alg_params)

    def compute_Q(self, agent, states):
        return agent._V(states).mean()

    def random_init(self, trial):
        n_features = trial.suggest_categorical('n_features', [32, 64])
        max_kl = trial.suggest_loguniform('max_kl', 1e-5, 1e-2)
        critic_lr = trial.suggest_loguniform('critic_lr', 1e-5, 1e-2)

        self.policy_params['n_features'] = n_features
        self.alg_params['max_kl'] = max_kl
        self.critic_params['n_features'] = n_features
        self.critic_params['optimizer']['params']['lr'] = critic_lr
    
    @classmethod
    def default(cls, critic_lr=3e-4, critic_network=Network, max_kl=1e-2, lam=.95, n_features=32, critic_fit_params=None, n_steps_per_fit=3000, n_epochs_cg=100, preprocessors=[StandardizationPreprocessor], use_cuda=False):

        policy_params = dict(
            std_0=1.,
            n_features=n_features,
            use_cuda=use_cuda)

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
            ent_coeff=0.0,
            max_kl=max_kl,
            lam=lam,
            n_epochs_line_search=10,
            n_epochs_cg=n_epochs_cg,
            cg_damping=1e-2,
            cg_residual_tol=1e-10,
            critic_fit_params=critic_fit_params,
            quiet=True)

        return cls(policy_params, critic_params, alg_params, n_steps_per_fit=n_steps_per_fit, preprocessors=preprocessors)