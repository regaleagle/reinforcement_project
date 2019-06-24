# pip install tf-nightly
#tf-nightly-gpu-2.0-preview - work

# pip install tf-agents-nightly
# pip install tfp-nightly
# pip install 'gym==0.10.11'
# pip install opencv-python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64

import tensorflow as tf

from tensorflow import TensorShape
from agents.tf_agents.agents.dqn import dqn_agent
from agents.tf_agents.drivers import dynamic_episode_driver
from agents.tf_agents.environments import suite_atari
from agents.tf_agents.environments import tf_py_environment
from agents.tf_agents.agents.sac import sac_agent
from agents.tf_agents.eval import metric_utils
from agents.tf_agents.metrics import tf_metrics
from atari_q_network import AtariQNetwork
from atari_q_network import AtariActorNetwork
from atari_q_network import AtariCriticNetwork
from agents.tf_agents.agents.ddpg import critic_network
from agents.tf_agents.agents.ddpg import actor_network
from agents.tf_agents.policies import random_tf_policy
from agents.tf_agents.replay_buffers import tf_uniform_replay_buffer
from agents.tf_agents.trajectories import policy_step
from agents.tf_agents.trajectories import trajectory
from agents.tf_agents.trajectories import time_step as ts
from agents.tf_agents.utils import common
from agents.tf_agents.networks import actor_distribution_network
from agents.tf_agents.networks import categorical_projection_network
from agents.tf_agents.networks import q_network
from time import time
import cv2
import PIL.Image
from ddq_agent import Ddq_Agent
from prox_pol_opt_agent import Ppo_Agent
from agents.tf_agents.metrics import tf_metric
from agents.tf_agents.replay_buffers import py_hashed_replay_buffer
from agents.tf_agents.environments import batched_py_environment
import time
from agents.tf_agents.environments import parallel_py_environment
import os
from absl import logging


class EpsilonMetric(tf_metric.TFStepMetric):
  """Counts the number of steps taken in the environment."""

  def __init__(self, epsilon, name='Epsilon', dtype=tf.float32):
    super(EpsilonMetric, self).__init__(name)
    self.epsilon = epsilon
    self.dtype = dtype
    self.epsilon_return = common.create_variable(
        initial_value=0, dtype=self.dtype, shape=(), name='environment_steps')

  def result(self):
    return tf.identity(
        self.epsilon_return, name=self.name)

  def call(self, trajectory):
    self.epsilon_return.assign(self.epsilon())
    return trajectory



def run(env_name, agent_type, root_dir="result_dir_", trial_num = None, n_step = 25):


    #params
    result_dir = root_dir + (trial_num if trial_num is not None else str(time.time() % 1000000))
    summary_interval = 10
    conv_layer_params = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
    fc_layer_params = (512,)
    num_iterations = 1000000
    target_update_period = 100
    epsilon_greedy = 0.1
    replay_buffer_capacity = 100000
    target_update_tau = 0.1

    collect_epsiodes_per_iteration = 20

    num_parallel_environments = 4

    use_tf_functions = True

    initial_collect_episodes = 1000
    num_environment_episodes = 1000000

    logging.set_verbosity(logging.INFO)


    tf.compat.v1.enable_v2_behavior()
    tf.enable_eager_execution()
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        result_dir, flush_millis=10000)
    train_summary_writer.set_as_default()

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):
        # atari_env = suite_atari.load(
        #     env_name,
        #     max_episode_steps=50000,
        #     gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)

        train_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(
                [lambda: suite_atari.load(
                env_name,
                max_episode_steps=50000,
                gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)] * num_parallel_environments))

        environment_episode_metric = tf_metrics.NumberOfEpisodes()
        step_metrics = [
            tf_metrics.EnvironmentSteps(),
            environment_episode_metric,
        ]
        if agent_type == 'ddqn':
            epsilon = tf.compat.v1.train.polynomial_decay(
                1.0,
                global_step,
                5000,
                end_learning_rate=epsilon_greedy)
            epsilon_metric = EpsilonMetric(epsilon=epsilon, name="Epsilon")
            agent = Ddq_Agent(convolutional_layers=conv_layer_params, target_update_tau=target_update_tau,
                              target_update_period=target_update_period, fully_connected_layers=fc_layer_params,
                              tf_env=train_env, n_step_update=n_step, global_step=global_step, epsilon_greedy=epsilon)
            train_metrics = step_metrics + [
                tf_metrics.AverageReturnMetric(),
                tf_metrics.AverageEpisodeLengthMetric(),
                epsilon_metric
            ]
        elif agent_type == 'ppo':
            agent = Ppo_Agent(convolutional_layers=conv_layer_params, fully_connected_layers=fc_layer_params,
                              tf_env=train_env, global_step=global_step)
            train_metrics = step_metrics + [
                tf_metrics.AverageReturnMetric(),
                tf_metrics.AverageEpisodeLengthMetric()
            ]
        else:
            raise ValueError('No appropriate agent found')

        agent.initialize()

        print("agent initialized")

        #policy
        eval_policy = agent.policy
        collect_policy = agent.collect_policy

        #define the buffer

        # py_observation_spec = train_env.observation_spec()
        # py_time_step_spec = ts.time_step_spec(py_observation_spec)
        # py_action_spec = policy_step.PolicyStep(train_env.action_spec())
        # data_spec = trajectory.from_transition(
        #     py_time_step_spec, py_action_spec, py_time_step_spec)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity)

        #create the driver (the thing that uses the policy and gets info)

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            train_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=collect_epsiodes_per_iteration)

        train_checkpointer = common.Checkpointer(
            ckpt_dir=result_dir,
            agent=agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(result_dir, 'policy'),
            policy=eval_policy,
            global_step=global_step)
        rb_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(result_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=replay_buffer)

        train_checkpointer.initialize_or_restore()
        rb_checkpointer.initialize_or_restore()

        if use_tf_functions:
            # To speed up collect use common.function.
            collect_driver.run = common.function(collect_driver.run)
            agent.train = common.function(agent.train)

        if agent_type == 'ddqn':

        # Collect initial replay data.
            logging.info(
                'Initializing replay buffer by collecting experience for %d steps with '
                'a random policy.', initial_collect_episodes)
            #might need to change for ppo
            initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
                train_env,
                agent.collect_policy,
                observers=[replay_buffer.add_batch],
                num_episodes=initial_collect_episodes)
            print("done")

            print("initial")
            initial_collect_driver.run()
            print("done")

        # results = metric_utils.eager_compute(
        #     eval_metrics,
        #     eval_tf_env,
        #     eval_policy,
        #     num_episodes=num_eval_episodes,
        #     train_step=global_step,
        #     summary_writer=eval_summary_writer,
        #     summary_prefix='Metrics',
        # )
        # if eval_metrics_callback is not None:
        #     eval_metrics_callback(results, global_step.numpy())
        # metric_utils.log_metrics(eval_metrics)

        # time_step = None
        # policy_state = collect_policy.get_initial_state(tf_env.batch_size)
        #
        # timed_at_step = global_step.numpy()
        # time_acc = 0
        #
        # # Dataset generates trajectories with shape [Bx2x...]
        # dataset = replay_buffer.as_dataset(
        #     num_parallel_calls=3,
        #     sample_batch_size=batch_size,
        #     num_steps=train_sequence_length + 1).prefetch(3)
        # iterator = iter(dataset)



        collect_time = 0
        train_time = 0
        timed_at_step = global_step.numpy()

        # environment_steps_metric = tf_metrics.EnvironmentSteps()
        # step_metrics = [
        #     tf_metrics.NumberOfEpisodes(),
        #     environment_steps_metric,
        # ]

        if agent_type == 'ddqn':
            time_step = None
            policy_state = collect_policy.get_initial_state(train_env.batch_size)

            timed_at_step = global_step.numpy()
            time_acc = 0

            dataset = replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=30,
                num_steps=n_step + 1).prefetch(3)
            iterator = iter(dataset)
            for _ in range(num_iterations):
                print(global_step.numpy())
                start_time = time.time()

                time_step, policy_state = collect_driver.run(
                    time_step=time_step,
                    policy_state=policy_state,
                )
                experience, _ = next(iterator)
                train_loss = agent.train(experience)
                time_acc += time.time() - start_time

                #TODO log_interval

                if global_step.numpy() % 200 == 0:
                    logging.info('step = %d, loss = %f', global_step.numpy(),
                                 train_loss.loss)
                    steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
                    logging.info('%.3f steps/sec', steps_per_sec)
                    tf.compat.v2.summary.scalar(
                        name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                    timed_at_step = global_step.numpy()
                    time_acc = 0

                for train_metric in train_metrics:
                    train_metric.tf_summaries(
                        train_step=global_step, step_metrics=step_metrics)

                if global_step.numpy() % 2000 == 0:
                    train_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % 2000 == 0:
                    policy_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % 2000 == 0:
                    rb_checkpointer.save(global_step=global_step.numpy())

                # if global_step.numpy() % eval_interval == 0:
                #     results = metric_utils.eager_compute(
                #         eval_metrics,
                #         eval_tf_env,
                #         eval_policy,
                #         num_episodes=num_eval_episodes,
                #         train_step=global_step,
                #         summary_writer=eval_summary_writer,
                #         summary_prefix='Metrics',
                #     )
                #     if eval_metrics_callback is not None:
                #         eval_metrics_callback(results, global_step.numpy())
                #     metric_utils.log_metrics(eval_metrics)

        else:
            collect_driver.run = common.function(collect_driver.run, autograph=False)
            agent.train = common.function(agent.train, autograph=False)
            collect_time = 0
            train_time = 0
            timed_at_step = global_step.numpy()
            while environment_episode_metric.result() < num_environment_episodes:
                global_step_val = global_step.numpy()
                print(global_step_val)
                # if global_step_val % eval_interval == 0:
                #     metric_utils.eager_compute(
                #         eval_metrics,
                #         eval_tf_env,
                #         eval_policy,
                #         num_episodes=num_eval_episodes,
                #         train_step=global_step,
                #         summary_writer=eval_summary_writer,
                #         summary_prefix='Metrics',
                #     )

                start_time = time.time()
                collect_driver.run()
                collect_time += time.time() - start_time

                start_time = time.time()
                trajectories = replay_buffer.gather_all()
                total_loss, _ = agent.train(experience=trajectories)
                replay_buffer.clear()
                train_time += time.time() - start_time
                print("train time", train_time)

                for train_metric in train_metrics:
                    train_metric.tf_summaries(
                        train_step=global_step, step_metrics=step_metrics)

                if global_step_val % 200 == 0:
                    logging.info('step = %d, loss = %f', global_step_val, total_loss)
                    steps_per_sec = (
                        (global_step_val - timed_at_step) / (collect_time + train_time))
                    logging.info('%.3f steps/sec', steps_per_sec)
                    logging.info('collect_time = {}, train_time = {}'.format(
                        collect_time, train_time))
                    with tf.compat.v2.summary.record_if(True):
                        tf.compat.v2.summary.scalar(
                            name='global_steps_per_sec', data=steps_per_sec, step=global_step)

                    timed_at_step = global_step_val
                    collect_time = 0
                    train_time = 0

        # One final eval before exiting.
        # metric_utils.eager_compute(
        #     eval_metrics,
        #     eval_tf_env,
        #     eval_policy,
        #     num_episodes=num_eval_episodes,
        #     train_step=global_step,
        #     summary_writer=eval_summary_writer,
        #     summary_prefix='Metrics',
        # )








if __name__ == '__main__':
  run('BreakoutDeterministic-v4', 'ppo')



# def compute_avg_return(environment, policy, num_episodes=5):
#
#   total_return = 0.0
#   for _ in range(num_episodes):
#
#     time_step = environment.reset()
#     episode_return = 0.0
#     print(time_step.is_last())
#     actions = []
#     while not time_step.is_last():
#       action_step = policy.action(time_step)
#       time_step = environment.step(action_step.action)
#       actions.append(action_step.action)
#       # im = PIL.Image.fromarray(environment._env.envs[-1]._env.render())
#       # im.show()
#       # im = plt.imshow(environment._env.envs[-1]._env.render())
#       # plt.show()
#       # cv2.imshow('image', environment._env.envs[-1]._env.render())
#       # cv2.waitKey(10)
#
#       episode_return += time_step.reward
#     total_return += episode_return
#
#   avg_return = total_return / num_episodes
#   print(actions)
#   return avg_return.numpy()[0]

# def run(env_name):
#     tf.compat.v1.enable_v2_behavior()
#     tf.enable_eager_execution()
#     batch_size = 30  # @param
#     train_summary_writer = tf.compat.v2.summary.create_file_writer(
#         "summary", flush_millis=10 * 1000)
#     train_summary_writer.set_as_default()
#
#     with tf.compat.v2.summary.record_if(
#             lambda: tf.math.equal(global_step % 10, 0)):
#
#
#
#         print("train env")
#         train_py_env = suite_atari.load(env_name,max_episode_steps=108000 / 4,
#               gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
#         train_env = tf_py_environment.TFPyEnvironment(train_py_env)
#         train_env.reset()
#         print("done")
#
#         print("first state")
#         state = train_env.reset()
#         print(state)
#
#         print("eval env")
#         eval_py_env = suite_atari.load(env_name,max_episode_steps=108000 / 4,
#               gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
#         eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
#         eval_env.reset()
#         print("done")
#
#         #the convolutional layers of the q_network
#         conv_layer_params = (
#             (32, (8, 8), 4), (64, (4, 4), 2))
#
#         #the fully connected layer/s of the q_network
#         fc_layer_params = (512,)
#
#         # actor_net = q_network.QNetwork(
#         #     train_env.observation_spec(),
#         #     train_env.action_spec(),
#         #     fc_layer_params=fc_layer_params, conv_layer_params=conv_layer_params)
#         #
#         # actor_net = AtariQNetwork(
#         #     train_env.observation_spec(),
#         #     train_env.action_spec(),
#         #     fc_layer_params=fc_layer_params, conv_layer_params=conv_layer_params)
#         #
#         # critic_net = AtariQNetwork(
#         #     train_env.observation_spec(),
#         #     train_env.action_spec(),
#         #     batch_squash=False,
#         #     fc_layer_params=fc_layer_params, conv_layer_params=conv_layer_params)
#         # actor_net = actor_network.ActorNetwork(
#         #     train_env.observation_spec(),
#         #     train_env.action_spec(),
#         #     fc_layer_params=fc_layer_params,
#         #     conv_layer_params=conv_layer_params)
#
#         discrete_projection_net = categorical_projection_network.CategoricalProjectionNetwork(
#             train_env.action_spec(),
#             logits_init_output_factor=0.1)
#
#
#         print("networks")
#         actor_net = AtariActorNetwork(
#             train_env.observation_spec(),
#             train_env.action_spec(),
#             fc_layer_params=fc_layer_params,
#             conv_layer_params=conv_layer_params
#             # ,
#             # discrete_projection_net=discrete_projection_net,
#             # continuous_projection_net=None
#             )
#         batchRank = TensorShape(30)
#
#         criticObserveSpec = train_env.observation_spec()
#         # print(criticObserveSpec.shape)
#         # criticObserveSpec._shape = batchRank.concatenate(train_env.observation_spec().shape)
#         # print(criticObserveSpec.shape)
#         # critic_net = q_network.QNetwork(
#         #     criticObserveSpec,
#         #     train_env.action_spec(),
#         #     fc_layer_params=fc_layer_params, conv_layer_params=conv_layer_params)
#
#         critic_net = AtariCriticNetwork(
#             (train_env.observation_spec(), train_env.action_spec()),
#             observation_fc_layer_params=fc_layer_params,
#             observation_conv_layer_params=conv_layer_params)
#
#         print("done")
#
#         num_iterations = 300000  # @param
#
#         initial_collect_steps = 1000  # @param
#         collect_steps_per_iteration = 1  # @param
#         replay_buffer_capacity = 100000  # @param
#
#         batch_size = 30  # @param
#
#         critic_learning_rate = 3e-3  # @param
#         actor_learning_rate = 3e-3  # @param
#         alpha_learning_rate = 3e-3  # @param
#         target_update_tau = 0.005  # @param
#         target_update_period = 1  # @param
#         gamma = 0.99  # @param
#         reward_scale_factor = 4.0  # @param
#         gradient_clipping = None  # @param
#
#         log_interval = 1000  # @param
#
#         num_eval_episodes = 1  # @param
#         eval_interval = 1000  # @param
#
#         print("global step")
#         global_step = tf.compat.v1.train.get_or_create_global_step()
#
#         print(global_step)
#         tf_agent = sac_agent.SacAgent(
#             train_env.time_step_spec(),
#             train_env.action_spec(),
#             actor_network=actor_net,
#             critic_network=critic_net,
#             actor_optimizer=tf.compat.v1.train.AdamOptimizer(
#                 learning_rate=actor_learning_rate),
#             critic_optimizer=tf.compat.v1.train.AdamOptimizer(
#                 learning_rate=critic_learning_rate),
#             alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
#                 learning_rate=alpha_learning_rate),
#             target_update_tau=target_update_tau,
#             target_update_period=target_update_period,
#             td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
#             target_entropy=4.0,
#             gamma=gamma,
#             reward_scale_factor=reward_scale_factor,
#             gradient_clipping=gradient_clipping,
#             debug_summaries=True,
#             summarize_grads_and_vars=True,
#             train_step_counter=global_step,
#         )
#         print("print(train_env.time_step_spec())")
#         print(train_env.time_step_spec())
#         print("print(train_env.action_spec()) ")
#         print(train_env.action_spec())
#         print("print(train_env.observation_spec())")
#         print(train_env.observation_spec())
#
#         print("tf.agent")
#         print(tf_agent.action_spec, tf_agent.time_step_spec)
#
#         print("criticnetwork")
#         print( critic_net)
#         print("actornetwork")
#         print(actor_net.input_tensor_spec)
#
#         print("initialize agent")
#         tf_agent.initialize()
#         print("done")
#
#
#
#         print("policies")
#         eval_policy = tf_agent.policy
#         collect_policy = tf_agent.collect_policy
#         print("done")
#
#         print("collect_policy")
#         print(collect_policy)
#
#         print("create replay buffer")
#         py_time_step_spec = ts.time_step_spec(train_env.observation_spec())
#         py_action_spec = policy_step.PolicyStep(train_env.action_spec())
#         data_spec = trajectory.from_transition(
#             py_time_step_spec, py_action_spec, py_time_step_spec)
#         replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#             data_spec=data_spec,
#             batch_size=train_env.batch_size,
#             max_length=replay_buffer_capacity)
#         print("done")
#
#         #Driver for collecting initial data
#         print("initial driver")
#         initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
#             train_env,
#             collect_policy,
#             observers=[replay_buffer.add_batch],
#             num_steps=initial_collect_steps)
#         print("done")
#
#         print("initial")
#         initial_collect_driver.run()
#         print("done")
#
#         #turn buffer to dataset ( define parallel here?)
#
#         dataset = replay_buffer.as_dataset(
#             num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
#
#         iterator = iter(dataset)
#
#         collect_driver = dynamic_step_driver.DynamicStepDriver(
#             train_env,
#             collect_policy,
#             observers=[replay_buffer.add_batch],
#             num_steps=collect_steps_per_iteration)
#
#         # (Optional) Optimize by wrapping some of the code in a graph using TF function.
#         tf_agent.train = common.function(tf_agent.train)
#         collect_driver.run = common.function(collect_driver.run)
#
#         # Reset the train step
#         tf_agent.train_step_counter.assign(0)
#
#         # Evaluate the agent's policy once before training.
#         avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
#         returns = [avg_return]
#         print(returns)
#         beg = time()
#
#         for _ in range(num_iterations):
#             # Collect a few steps using collect_policy and save to the replay buffer.
#             for _ in range(collect_steps_per_iteration):
#                 collect_driver.run()
#
#             # Sample a batch of data from the buffer and update the agent's network.
#             experience, unused_info = next(iterator)
#             train_loss = tf_agent.train(experience)
#
#             step = tf_agent.train_step_counter.numpy()
#             # steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
#             # tf.compat.v2.summary.scalar(
#             #     name='global_steps_per_sec', data=steps_per_sec, step=global_step)
#
#             if step % log_interval == 0:
#                 end = time()
#                 print("time per 500 step: ", (end - beg))
#                 beg = time()
#                 print('step = {0}: loss = {1}'.format(step, train_loss.loss))
#
#             if step % eval_interval == 0:
#                 avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
#                 print('step = {0}: Average Return = {1}'.format(step, avg_return))
#                 returns.append(avg_return)