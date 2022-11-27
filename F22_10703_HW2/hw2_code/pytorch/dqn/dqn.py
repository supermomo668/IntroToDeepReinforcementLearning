#!/usr/bin/env python
from email import policy
from re import L
import numpy as np, gym, sys, copy, argparse
import os
import torch
import random
from collections import deque, namedtuple
import tqdm
import matplotlib.pyplot as plt

class FullyConnectedModel(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, 16)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 16)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(16, 16)
        self.activation3 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(16, output_size)
        #no activation output layer

        #initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

        self.input_size = input_size

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs)
        inputs = inputs.reshape(-1, self.input_size)
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_layer(x)
        return x


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, lr, logdir=None, init=None):
    # Define your network architecture here. It is also a good idea to define any training operations
    # and optimizers here, initialize your variables, or alternately compile your model here.
        n_S = len(env.reset())
        n_A = env.action_space.n
        self.model = FullyConnectedModel(n_S, n_A)
        self.logdir = logdir if logdir else "/home/junli/CMU/10703/F22_10703_Homework_2/hw2_code/pytorch/dqn/ckpt"
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.MSE = torch.nn.MSELoss()

        if init:
            self.reinit(init)

    def save_model_weights(self, suffix):
    # Helper function to save your model / weights.
        path = os.path.join(self.logdir, f"model_{suffix}")
        # if not os.path.exists(path):
        #     os.makedirs(path)
        torch.save(self.model.state_dict(), path)
        return path

    def load_model(self, model_file):
    # Helper function to load an existing model.
        return self.model.load_state_dict(torch.load(model_file))

    def load_model_weights(self,weight_file):
    # Optional Helper function to load model weights.
        pass

    def reinit(self, state_dict):
        self.model.load_state_dict(state_dict)
        


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

    # The memory essentially stores transitions recorder from the agent
    # taking actions in the environment.

    # Burn in episodes define the number of episodes that are written into the memory from the
    # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
    # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

    # Hint: you might find this useful:
    # 		collections.deque(maxlen=memory_size)
        self.burn_in = burn_in
        self.buffer = deque(maxlen=memory_size)
        self.transition_wapper = namedtuple("Transition", 
                                                ("state", "action", "reward", "next_state", "eps_done")
                                            )
    
    def sample_batch(self, batch_size=32):
    # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
    # You will feed this to your model to train.
        return random.sample(self.buffer, batch_size)

    def append(self, transition):
    # Appends transition to the memory.
        self.buffer.appendleft(self.transition_wapper(*transition))


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, lr, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env = gym.make(environment_name)
        self.environment_name = environment_name
        self.gamma = 0.99
        self.epsilon = 0.05
        self.E = 200
        self.batch_size = 32

        self.policyQ = QNetwork(self.env, lr)
        self.targetQ = QNetwork(self.env, lr, init=self.policyQ.model.state_dict())
        self.testQ = QNetwork(self.env, lr)
        self.testQ.model.eval()
        self.targetQ.model.eval()
        

        self.replay = Replay_Memory()
        self.benchmark = []

    
    @torch.no_grad()
    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.

        random_policy = np.random.binomial(1, self.epsilon)
        if random_policy:
            return np.random.randint(2)
        
        # with torch.no_grad():
        return torch.argmax(q_values, 1).item()

    @torch.no_grad()
    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        # with torch.no_grad():
        return torch.argmax(q_values, 1).item()

    def train(self):
        # In this function, we will train our network.
        self.burn_in_memory()
        c = 0
        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        for iter in range(1, self.E+1):
            state = self.env.reset() # re init the epsoid
            eps_done = False
            
            while not eps_done:
                # with torch.no_grad():
                action = self.epsilon_greedy_policy(self.policyQ.model(state))

                new_state, rewards, eps_done, info = self.env.step(action)
                self.replay.append( (state, action, rewards, new_state, eps_done) )
                state = new_state

                batch_data = self.replay.sample_batch(self.batch_size)
                batch_data = self.replay.transition_wapper(*zip(*batch_data))

                batch_state = torch.tensor(batch_data.state)
                batch_action = torch.tensor(batch_data.action)
                batch_next_state = torch.tensor(batch_data.next_state)
                batch_rewards = torch.tensor(batch_data.reward)
                not_done_mask = torch.tensor([not done for done in batch_data.eps_done])

                q_values = self.policyQ.model(batch_state) \
                                        .gather(1, batch_action.reshape(self.batch_size, 1))

                q_next_state = torch.zeros(self.batch_size)
                q_next_state[not_done_mask] = self.targetQ.model(batch_next_state[not_done_mask]).max(1)[0]

                V = batch_rewards + self.gamma * q_next_state

                loss = self.policyQ.MSE(V, q_values.squeeze(1))

                self.policyQ.optimizer.zero_grad()
                loss.backward()
                self.policyQ.optimizer.step()

                if c % 50 == 0:
                    self.targetQ.reinit(self.policyQ.model.state_dict())
                print(f"Loss @ eps {iter} step {c}: {loss}")
                c += 1

            if iter % 10 == 0:
                self.benchmark += [self.test(model_file=self.policyQ.model.state_dict())]

            # if iter % int(self.E/3) == 0:
          	#     test_video(self, self.environment_name, iter)

        print(self.benchmark)
        self.policyQ.save_model_weights("200.pt")



    def test(self, model_file=None):
        # Evaluate the performance of your agent over 20 episodes, by calculating average cumulative rewards (returns) for the 20 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using replay memory.
        self.testQ.reinit(model_file)
        mean_rewards = []
        test_ep = 20
        for i in range(test_ep):
            eps_done = False
            state = self.env.reset()
            r = 0
            while not eps_done:
                # with torch.no_grad():
                action = self.greedy_policy(self.testQ.model(state))
                new_state, rewards, eps_done, info = self.env.step(action)
                # print(rewards)
                state = new_state
                r += rewards
            mean_rewards += [r]
        return np.mean(np.array(mean_rewards))

        

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        n_iter = self.replay.burn_in
        eps_done = True
        iter = 0
        while iter < n_iter:
            if eps_done:
                state = self.env.reset() # reset game to new epsoid
            iter += 1
            random_action = self.env.action_space.sample() # take rand action
            new_state, reward, eps_done, info = self.env.step(random_action)
            # new_state = None if eps_done else new_state

            trans = (state, random_action, reward, new_state, eps_done)
            self.replay.append(trans)

            state = new_state



# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop.
def test_video(agent, env, epi):
    # Usage:
    # 	you can pass the arguments within agent.train() as:
    # 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(agent.policyQ.model(state))
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str,default='CartPole-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4)
    return parser.parse_args()


def main(args):

    args = parse_arguments()
    environment_name = args.env
    lr = args.lr
    render = args.render

    returns = []
    for _ in range(5):
        agent = DQN_Agent(environment_name, lr, render)
        agent.train()
        returns += [agent.benchmark]
        
    # for i in range(5):
        # print(len(returns[i]))
    returns = np.array(returns)
    returns = np.stack(returns, 0)
    x = list(range(0, 200, 10))
    plt.figure()
    plt.plot(x, np.mean(returns,axis=0), label="Mean")
    plt.plot(x, np.min(returns,axis=0), label="Min")
    plt.plot(x, np.max(returns,axis=0), label="Max")

    plt.xlabel("Episode")
    plt.ylabel("Aggregation of Returns")
    plt.title("Aggregations of Trials' Mean Test Returns")

    plt.legend()

    
    plt.savefig("/home/junli/CMU/10703/F22_10703_Homework_2/hw2_code/pytorch/dqn/2.2.png")
    plt.show()

    # You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main(sys.argv)