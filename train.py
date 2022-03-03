from abc import ABC

import ray
from ray import tune
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf import TFModelV2, FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from ray.tune import register_env

from BriscolaChiamata import BriscolaChiamataEnv

tf1, tf, tfv = try_import_tf()

class ParametricActionsModel(TFModelV2):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)
        orig_obs_space = obs_space.original_space.spaces['observation']
        self.action_embed_model = FullyConnectedNetwork(
            orig_obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_action_embedding"
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model({
            "obs": input_dict["obs"]['observation']
        })

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


def env_creator(env_config):
    return BriscolaChiamataEnv()  # return an env instance


def briscolaMain():
    ray.init(local_mode=True)
    env = env_creator({})
    # ray.rllib.utils.check_env(env)
    register_env("BriscolaChiamata-v0", lambda config: PettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
    tune.run("PPO",
             config={"env": "BriscolaChiamata-v0",
                     "model": {
                         "custom_model": "pa_model"
                     },
                     "evaluation_interval": 2,
                     "evaluation_duration": 20,
                     "num_gpus": 0,
                     # "multiagent": {
                     #     "policies": set(env.agents),
                     #     "policy_mapping_fn": (lambda agent_id, episode, **kwargs: agent_id),
                     # },
                     },
             local_dir="BriscolaChiamata-v0",
             checkpoint_freq=2,
             #resume=True # Uncomment when doing actual experiments
             )


def cartpoleMain():
    ray.init()
    tune.run("PPO",
             config={"env": "CartPole-v1",
                     "evaluation_interval": 2,
                     "evaluation_duration": 20,
                     "num_gpus": 0
                     },
             local_dir="cartpole_v1"
             )


if __name__ == "__main__":
    briscolaMain()
    # cartpoleMain()
