from agents.tf_agents.networks import q_network
from agents.tf_agents.networks import actor_distribution_network
from agents.tf_agents.agents.ddpg import critic_network
import tensorflow as tf

class AtariQNetwork(q_network.QNetwork):
  """QNetwork subclass that divides observations by 255."""

  def call(self, observation, step_type=None, network_state=None):
    state = tf.cast(observation, tf.float32)
    # We divide the grayscale pixel values by 255 here rather than storing
    # normalized values beause uint8s are 4x cheaper to store than float32s.
    state = state / 255
    return super(AtariQNetwork, self).call(
        state, step_type=step_type, network_state=network_state)

class AtariActorNetwork(actor_distribution_network.ActorDistributionNetwork):
  """QNetwork subclass that divides observations by 255."""

  def call(self, observation, step_type=None, network_state=None):
    state = tf.cast(observation, tf.float32)
    # We divide the grayscale pixel values by 255 here rather than storing
    # normalized values beause uint8s are 4x cheaper to store than float32s.
    state = state / 255
    return super(AtariActorNetwork, self).call(
        state, step_type=step_type, network_state=network_state)

class AtariCriticNetwork(critic_network.CriticNetwork):
  """QNetwork subclass that divides observations by 255."""

  def call(self, input, step_type=None, network_state=None):
    observation, action = input
    state = tf.cast(observation, tf.float32)
    # We divide the grayscale pixel values by 255 here rather than storing
    # normalized values beause uint8s are 4x cheaper to store than float32s.
    state = state / 255
    return super(AtariCriticNetwork, self).call(
      (state, action), step_type=step_type, network_state=network_state)
