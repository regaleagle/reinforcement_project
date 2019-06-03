# pip install tf-nightly
# pip install tf-agents-nightly
# pip install tfp-nightly
# pip install 'gym==0.10.11'
# pip install opencv-python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_atari
from tf_agents.environments import tf_py_environment
from tf_agents.agents.sac import sac_agent
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from atari_q_network import AtariQNetwork
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import actor_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.networks import actor_distribution_network
from time import time


def compute_avg_return(environment, policy, num_episodes=5):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


def run(env_name):
    tf.compat.v1.enable_v2_behavior()

    print("train env")
    train_py_env = suite_atari.load(env_name,max_episode_steps=108000 / 4,
          gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    train_env.reset()
    print("done")

    print("eval env")
    eval_py_env = suite_atari.load(env_name,max_episode_steps=108000 / 4,
          gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    eval_env.reset()
    print("done")

    #the convolutional layers of the q_network
    conv_layer_params = (
        (32, (8, 8), 4), (64, (4, 4), 2))

    #the fully connected layer/s of the q_network
    fc_layer_params = (512,)

    # q_net_actor = AtariQNetwork(
    #     train_env.observation_spec(),
    #     train_env.action_spec(),
    #     fc_layer_params=fc_layer_params, conv_layer_params=conv_layer_params)

    # q_net_critic = AtariQNetwork(
    #     train_env.observation_spec(),
    #     train_env.action_spec(),
    #     fc_layer_params=fc_layer_params, conv_layer_params=conv_layer_params)
    # actor_net = actor_network.ActorNetwork(
    #     train_env.observation_spec(),
    #     train_env.action_spec(),
    #     fc_layer_params=fc_layer_params,
    #     conv_layer_params=conv_layer_params)

    print("networks")
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params,
        conv_layer_params=conv_layer_params)


    critic_net = critic_network.CriticNetwork(
        (train_env.observation_spec(), train_env.action_spec()),
        observation_fc_layer_params=fc_layer_params,
        observation_conv_layer_params=conv_layer_params)

    print("done")

    num_iterations = 300000  # @param

    initial_collect_steps = 10000  # @param
    collect_steps_per_iteration = 1  # @param
    replay_buffer_capacity = 100000  # @param

    batch_size = 30  # @param

    critic_learning_rate = 3e-4  # @param
    actor_learning_rate = 3e-4  # @param
    alpha_learning_rate = 3e-4  # @param
    target_update_tau = 0.005  # @param
    target_update_period = 1  # @param
    gamma = 0.99  # @param
    reward_scale_factor = 1.0  # @param
    gradient_clipping = None  # @param

    log_interval = 500  # @param

    num_eval_episodes = 1  # @param
    eval_interval = 500  # @param

    print("global step")
    global_step = tf.compat.v1.train.get_or_create_global_step()

    print("done")
    tf_agent = sac_agent.SacAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        train_step_counter=global_step)

    print("initialize agent")
    tf_agent.initialize()
    print("done")

    print("policies")
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy
    print("done")

    print("create replay buffer")
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)
    print("done")

    #Driver for collecting initial data
    print("initial driver")
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps)
    print("done")

    print("initial")
    initial_collect_driver.run()
    print("done")

    #turn buffer to dataset ( define parallel here?)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)

    iterator = iter(dataset)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)
    collect_driver.run = common.function(collect_driver.run)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]
    print(returns)
    beg = time()

    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_driver.run()

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = tf_agent.train(experience)

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            end = time()
            print("time per 500 step: ", (end - beg))
            beg = time()
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

if __name__ == '__main__':
  run('BreakoutDeterministic-v4')