from enum import Enum

class Actions(Enum):
    NONE = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    ATTACK_MELEE_UP = 5
    ATTACK_MELEE_DOWN = 6
    ATTACK_MELEE_LEFT = 7
    ATTACK_MELEE_RIGHT = 8
    ATTACK_RANGED_UP = 9
    ATTACK_RANGED_DOWN = 10
    ATTACK_RANGED_LEFT = 11
    ATTACK_RANGED_RIGHT = 12
    DASH_UP = 13
    DASH_DOWN = 14
    DASH_LEFT = 15
    DASH_RIGHT = 16
    ABSOLUTE_DEFENSE_ABILITY = 17
    ATTACK_BOOST_ABILITY = 18

class AlgoType(Enum):
    BASIC_ML = 0
    Q_VALUE = 1
    POLICY_GRADIENT = 2

class AlgoName(Enum):
    DQN = 'dqn'
    N_STEP_DQN = 'nstep_dqn'
    DOUBLE_DQN = 'double_dqn'
    DUELING_DQN = 'dueling_dqn'
    PER_DQN = 'per_dqn'
    NOISY_DQN = 'noisy_dqn'
    DISTRIBUTIONAL_DQN = 'distributional_dqn'
    RAINBOW_DQN = 'rainbow_dqn'
    REINFORCE ='reinforce'
    A2C ='a2c'
    PPO = 'ppo'
    DECISION_TREE = 'decision_tree'

class ActivationFunction(Enum):
    RELU = 'relu'
    TANH = 'tanh'
    SILU = 'silu'

class ProjectileDirection(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class AgentType(Enum):
    BASIC = 0
    RL = 1

class SignificantStatus(Enum):
    NOT_SIGNIFICANT = 0
    SIGNIFICANT = 1
    VERY_SIGNIFICANT = 2

class EnvType(Enum):
    TRAIN = 0
    TEST = 1

class EnemyType(Enum):
    BASIC = 0
    PREVIOUS_MODEL = 1


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3