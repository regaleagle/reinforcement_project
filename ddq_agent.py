

from atari_q_network import AtariQNetwork
from tf_agents.agents.dqn import dqn_agent
import tensorflow as tf


class Ddq_Agent(dqn_agent.DdqnAgent):
    def __init__(self,
                 convolutional_layers,
                 fully_connected_layers,
                 tf_env,
                 global_step,
                 epsilon_greedy = 0.1,
                 n_step_update = 25):

        q_net = AtariQNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            fc_layer_params=fully_connected_layers, conv_layer_params=convolutional_layers)


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

        super(Ddq_Agent, self).__init__(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=epsilon_greedy,
            n_step_update=n_step_update,
            optimizer=tf.compat.v1.train.RMSPropOptimizer(2.5e-4),
            td_errors_loss_fn=dqn_agent.element_wise_huber_loss,
            gamma=0.9,
            # gradient_clipping=gradient_clipping, #investigate gradients and consider clipping
            train_step_counter=global_step)