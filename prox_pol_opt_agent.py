

from atari_actor_network import AtariActorNetwork
from agents.tf_agents.agents.ppo import ppo_agent
from agents.tf_agents.networks import value_network
import tensorflow as tf


class Ppo_Agent(ppo_agent.PPOAgent):
    def __init__(self,
                 convolutional_layers,
                 fully_connected_layers,
                 tf_env,
                 global_step,
                 num_epochs=25,
                 entropy_regularization=0.2):

        value_net = value_network.ValueNetwork(
            tf_env.observation_spec(), conv_layer_params=convolutional_layers, fc_layer_params=fully_connected_layers)

        actor_net = AtariActorNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            fc_layer_params=fully_connected_layers, conv_layer_params=convolutional_layers)

        optimiser = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)


        # self.agent = tf_agent = dqn_agent.DdqnAgent(
        #         tf_env.time_step_spec(),
        #         tf_env.action_spec(),
        #         q_network=q_net,
        #         epsilon_greedy=epsilon_greedy,
        #         n_step_update=n_step_update,
        #         optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
        #         td_errors_loss_fn=dqn_agent.element_wise_squared_loss,
        #         gamma=gamma,
        #         reward_scale_factor=reward_scale_factor,
        #         gradient_clipping=gradient_clipping,
        #         debug_summaries=debug_summaries,
        #         summarize_grads_and_vars=summarize_grads_and_vars,
        #         train_step_counter=global_step)
        #     tf_agent.initialize()

        super(Ppo_Agent, self).__init__(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            optimiser,
            actor_net=actor_net,
            value_net=value_net,
            num_epochs=num_epochs,
            train_step_counter=global_step,
            entropy_regularization=entropy_regularization)