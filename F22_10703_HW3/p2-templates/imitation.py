import numpy as np

## TensorFlow Only ##
import tensorflow as tf
import keras
from model_tensorflow import make_model


## Pytorch Only ##
# import torch
# from model_pytorch import make_model, ExpertModel


def action_to_one_hot(env, action):
    action_vec = np.zeros(env.action_space.n)
    action_vec[action] = 1
    return action_vec    

def generate_episode(env, policy):
    """Collects one rollout from the policy in an environment. The environment
    should implement the OpenAI Gym interface. A rollout ends when done=True. The
    number of states and actions should be the same, so you should not include
    the final state when done=True.

    Args:
    env: an OpenAI Gym environment.
    policy: The output of a deep neural network
    Returns:
    states: a list of states visited by the agent.
    actions: a list of actions taken by the agent. For tensorflow, it will be 
        helpful to use a one-hot encoding to represent discrete actions. The actions 
        that you return should be one-hot vectors (use action_to_one_hot()).
        For Pytorch, the Cross-Entropy Loss function will integers for action
        labels.
    rewards: the reward received by the agent at each step.
    """
    done = False
    state = env.reset()

    states = []
    actions = []
    rewards = []
    while not done:
        # WRITE CODE HERE
        

class Imitation():
    
    def __init__(self, env, num_episodes, expert_file):
        self.env = env
        
        # TensorFlow Only #
        self.expert = tf.keras.models.load_model(expert_file)
        
        # Pytorch Only #
        # self.expert = ExpertModel()
        # self.expert.load_state_dict(torch.load(expert_file))
        # self.expert.eval()
        
        self.num_episodes = num_episodes
        
        self.model = make_model()
        
    def generate_behavior_cloning_data(self):
        self._train_states = []
        self._train_actions = []
        for _ in range(self.num_episodes):
            states, actions, _ = generate_episode(self.env, self.expert)
            self._train_states.extend(states)
            self._train_actions.extend(actions)
        self._train_states = np.array(self._train_states)
        self._train_actions = np.array(self._train_actions)
        
    def generate_dagger_data(self):
        # WRITE CODE HERE
        # You should collect states and actions from the student policy
        # (self.model), and then relabel the actions using the expert policy.
        # This method does not return anything.
        # END
        
    def train(self, num_epochs=1, batch_size=64):
        """
        Train the model on data generated by the expert policy.
        Use Cross-Entropy Loss and a batch size of 64 when
        performing updates.
        Args:
            num_epochs: number of epochs to train on the data generated by the expert.
        Return:
            loss: (float) final loss of the trained policy.
            acc: (float) final accuracy of the trained policy
        """
        # WRITE CODE HERE
        # END
        return loss, acc


    def evaluate(self, policy, n_episodes=50):
        rewards = []
        for i in range(n_episodes):
            _, _, r = generate_episode(self.env, policy)
            rewards.append(sum(r))
        r_mean = np.mean(rewards)
        return r_mean
    