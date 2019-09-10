# pip install tf-nightly
#tf-nightly-gpu-2.0-preview - work

# pip install tf-agents-nightly
# pip install tfp-nightly
# pip install 'gym==0.10.11'
# pip install opencv-python
# pip install imageio

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from agents.tf_agents.drivers import dynamic_episode_driver
from agents.tf_agents.environments import suite_atari
from agents.tf_agents.environments import tf_py_environment
from agents.tf_agents.eval import metric_utils
from agents.tf_agents.metrics import tf_metrics
from agents.tf_agents.replay_buffers import tf_uniform_replay_buffer
from agents.tf_agents.utils import common
from time import time
from ddq_agent import Ddq_Agent
from parameters import Parameters
from prox_pol_opt_agent import Ppo_Agent
from agents.tf_agents.metrics import tf_metric
import time
from datetime import datetime
from agents.tf_agents.environments import parallel_py_environment
import os
from absl import logging
import imageio

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



def run(env_name, agent_type, root_dir="result_dir_", trial_num = None, n_step = 25, exploration_min=0.1, parameters = Parameters()):

    logging.set_verbosity(logging.INFO)

    tf.compat.v1.enable_v2_behavior()
    #tf.enable_eager_execution()

    ### Params ###

    result_dir = root_dir + agent_type + "_" + (trial_num if trial_num is not None else datetime.today().strftime('%Y-%m-%d'))
    summary_interval = parameters.summary_interval
    conv_layer_params = parameters.conv_layer_params
    fc_layer_params = parameters.fc_layer_params
    target_update_period = parameters.target_update_period
    exploration_min = exploration_min
    replay_buffer_capacity = parameters.replay_buffer_capacity
    target_update_tau = parameters.target_update_tau
    collect_episodes_per_iteration = parameters.collect_episodes_per_iteration
    num_parallel_environments = parameters.num_parallel_environments
    use_tf_functions = parameters.use_tf_functions
    initial_collect_episodes = parameters.initial_collect_episodes
    log_interval = parameters.log_interval
    checkpoint_interval = parameters.checkpoint_interval



    ### TensorBoard summary settings ###

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        result_dir, flush_millis=10000)
    train_summary_writer.set_as_default()

    global_step = tf.compat.v1.train.get_or_create_global_step()

    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):

        ### Training Environment setup ###

        train_env = tf_py_environment.TFPyEnvironment(
            parallel_py_environment.ParallelPyEnvironment(
                [lambda: suite_atari.load(
                env_name,
                max_episode_steps=50000,
                gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)] * num_parallel_environments))

        eval_py_env = suite_atari.load(
                env_name,
                max_episode_steps=50000,
                gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        environment_episode_metric = tf_metrics.NumberOfEpisodes()
        step_metrics = [
            tf_metrics.EnvironmentSteps(),
            environment_episode_metric,
        ]

        ### Agent specific setup ##

        if agent_type == 'ddqn':

            #Epsilon decay
            epsilon = tf.compat.v1.train.polynomial_decay(
                learning_rate=1.0,
                global_step=global_step,
                decay_steps=10000, #5000 for experiment
                end_learning_rate=exploration_min)

            epsilon_metric = EpsilonMetric(epsilon=epsilon, name="Epsilon")

            agent = Ddq_Agent(convolutional_layers=conv_layer_params, target_update_tau=target_update_tau,
                              target_update_period=target_update_period, fully_connected_layers=fc_layer_params,
                              tf_env=train_env, n_step_update=n_step, global_step=global_step, epsilon_greedy=epsilon)
            # Metrics for Tensorboard
            train_metrics = step_metrics + [
                tf_metrics.AverageReturnMetric(),
                tf_metrics.AverageEpisodeLengthMetric(),
                epsilon_metric
            ]
        elif agent_type == 'ppo':

            agent = Ppo_Agent(convolutional_layers=conv_layer_params, fully_connected_layers=fc_layer_params,
                              tf_env=train_env, global_step=global_step, entropy_regularization=exploration_min)
            # Metrics for Tensorboard
            train_metrics = step_metrics + [
                tf_metrics.AverageReturnMetric(),
                tf_metrics.AverageEpisodeLengthMetric()
            ]
        else:
            raise ValueError('No appropriate agent found')

        eval_metrics = [
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric()
        ]

        agent.initialize()

        print("agent initialized")

        # Define policy - eval will choose optimal steps, collect is for training and has exploration
        eval_policy = agent.policy
        collect_policy = agent.collect_policy

        # Define the buffer

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            agent.collect_data_spec,
            batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity)

        # Create the driver (the object that uses the policy to interact
        # with the Environment and generates data to train with)

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            train_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=collect_episodes_per_iteration)
        eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            eval_env,
            eval_policy,
            observers=eval_metrics,
            num_episodes=10)

        # Checkpoints for model and data saving

        train_checkpointer = common.Checkpointer(
            ckpt_dir=result_dir,
            agent=agent,
            global_step=global_step,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))

        policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(result_dir, 'policy'),
            policy=eval_policy,
            global_step=global_step)

        train_checkpointer.initialize_or_restore()
        policy_checkpointer.initialize_or_restore()

        if use_tf_functions:
            # To speed up collect use common.function.
            collect_driver.run = common.function(collect_driver.run)
            agent.train = common.function(agent.train)
            eval_driver.run = common.function(collect_driver.run)

        ### Data collection run for off-policy agents (currently supporting DDQN
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

            initial_collect_driver.run()

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
            end_time = time.time() + (12 * 60 * 60)
            while time.time() < end_time:
                start_time = time.time()

                time_step, policy_state = collect_driver.run(
                    time_step=time_step,
                    policy_state=policy_state,
                )
                experience, _ = next(iterator)
                train_loss = agent.train(experience)
                time_acc += time.time() - start_time

                if global_step.numpy() % log_interval == 0:
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

        elif agent_type == 'ppo':
            collect_driver.run = common.function(collect_driver.run, autograph=False)
            #eval_driver = common.function(eval_driver.run, autograph=False)
            agent.train = common.function(agent.train, autograph=False)
            collect_time = 0
            train_time = 0
            timed_at_step = global_step.numpy()
            end_time = time.time() + (12 * 60 * 60)
            while time.time() < end_time:
                global_step_val = global_step.numpy()
                start_time = time.time()
                collect_driver.run()
                collect_time += time.time() - start_time

                start_time = time.time()
                trajectories = replay_buffer.gather_all()
                total_loss, _ = agent.train(experience=trajectories)
                replay_buffer.clear()
                train_time += time.time() - start_time

                for train_metric in train_metrics:
                    train_metric.tf_summaries(
                        train_step=global_step, step_metrics=step_metrics)

                if global_step_val % log_interval == 0:
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
                if global_step.numpy() % checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step.numpy())

        train_checkpointer.save(global_step=global_step.numpy())
        policy_checkpointer.save(global_step=global_step.numpy())
        num_episodes = 3
        video_filename = result_dir + '/finalVid_ppo_0_1.mp4'
        with imageio.get_writer(video_filename, fps=60) as video:
            for _ in range(num_episodes):
                time_step = eval_py_env.reset()
                video.append_data(eval_py_env.render())
                counter = 5
                while counter > 0:
                    action_step = agent.policy.action(time_step)
                    time_step = eval_py_env.step(action_step.action)
                    viddata = eval_py_env.render()
                    video.append_data(viddata)
                    video.append_data(viddata)
                    video.append_data(viddata)
                    video.append_data(viddata)
                    video.append_data(viddata)
                    if time_step.is_last():
                        eval_py_env.step([1])
                        counter -= 1

if __name__ == '__main__':
  run(env_name='BreakoutDeterministic-v4',agent_type='ppo', trial_num='2019-09-09', exploration_min=0.1, parameters=Parameters(
      summary_interval=10,
      conv_layer_params=((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)),
      fc_layer_params=(512,),
      num_iterations=1000000,
      target_update_period=100,
      replay_buffer_capacity=100000,
      target_update_tau=0.1,
      collect_episodes_per_iteration=10,
      num_parallel_environments=4,
      use_tf_functions=True,
      initial_collect_episodes=1000,
      num_environment_episodes=1000000,
      log_interval=200,
      eval_interval=2000,
      checkpoint_interval=2000
  ))
