import sys, os, tqdm
import argparse
from pathlib import Path
import copy

import numpy as np, torch, gym
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from .net import NeuralNet
#
from gym.wrappers import Monitor  # , record_video,
from pyvirtualdisplay import Display
from IPython.display import HTML
from IPython import display as ipythondisplay

global DEBUG
device = 'cuda' if torch.cuda.is_available() else 'cpu'
proj_folder = Path('.').absolute()
DEBUG=False
print(f"Using device:{device}")
"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""
def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")
    

def wrap_env(env, save_path=proj_folder/'video'):
    save_path.mkdir(parents=True, exist_ok=True)
    # env = record_video.RecordVideo(env, proj_folder/'video', name_prefix='eval')
    env = Monitor(env, proj_folder/'video', video_callable=False ,force=True)
    return env
    
class A2C(object):
    # Implementation of N-step Advantage Actor Critic.

    def __init__(self, actor, actor_lr, N, nA, critic, critic_lr, baseline=False, a2c=True):
        # Note: baseline is true if we use reinforce with baseline
        #       a2c is true if we use a2c else reinforce
        # TODO: Initializes A2C.
        self.all_types = ['Reinforce','Baseline','A2C']
        self.type = 2 if a2c else (1 if baseline else 0)  # Pick one of: "A2C", "Baseline", "Reinforce"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor.to(device)
        # define run variabless
        if self.type == 2 or self.type == 1:   # Baseline or A2C
            self.critic = critic
            if self.type == 1:   # Baseline
                self.critic.to(device)
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
            else:  # A2C
                self.N = N
        else:   # Reinforce
            pass
        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        assert self.type is not None, "Type must be provided"

    def reinforce_criterion(self, y_action_logprob, y_target):
        loss = - torch.multiply(y_action_logprob, y_target)    # shape (t,)
        return torch.mean(loss)
    
    def baseline_criterion(self, y_pred, y_true):
        return torch.mean(torch.square(y_true-y_pred))
        
    def fit_model(self, y_action, y_target, optimizer, criterion, epochs=1, batch_size=1):
        """
        Fit x (states) -> y (expected sum of reward/values)
        """
        self.actor.train()
        n_example = len(y_action)
        history = dict.fromkeys(['loss'],[])
        for e in range(epochs):
            for i in range(int(np.ceil(n_example/batch_size))):
                start_idx, end_idx = batch_size*i, min(batch_size*(i+1), n_example)
                y_action_, y_target_ = y_action[start_idx: end_idx], y_target[start_idx: end_idx]
                loss = criterion(y_action_, y_target_)
                # # measure metrics and record loss
                history['loss'].append(loss)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                if DEBUG: print(f"Latest Loss:{history['loss'][-1]}")
        return history

    def evaluate_policy(self, env):
        # TODO: Compute Accumulative trajectory reward(set a trajectory length threshold if you want)
        """ compute rewards
        """
        self.actor.eval()
        _, _, r, _ = self.generate_episode(env)
        rtot = np.sum(np.array(r))
        return rtot

    def generate_episode(self, env, render=False):
        """
		# Generates an episode by executing the current policy in the given env.
		# Returns:
		# - a list of states, indexed by time step
            shape: (t)
		# - a list of actions, indexed by time step
            shape: (t, nA)
		# - a list of rewards, indexed by time step
            shape: (t)
        """
        states, actions, rewards, action_logprobs = [], [] ,[], []
        # 
        nS = env.observation_space.shape[0]
        nA = env.action_space.n
		# Start episode
        states.append(np.expand_dims(env.reset(), axis=0))
        terminal = False
        cts = 0
        eps = 1e-7
        while not terminal:
            cts+=1
            if render: env.render()
            ac = self.actor(torch.Tensor(states[-1]).to(device)).squeeze(0)   # ensure [nA] vector
            if DEBUG: print(f"Input state:{states[-1]}\nOutput action:{ac}")
            # cateogircal distribution
            action_dist = Categorical(torch.clamp(ac.flatten(), eps, 1-eps))
            curr_ac = action_dist.sample()  # Current Action
            ac_logprob = action_dist.log_prob(curr_ac)
            action_OH = np.eye(nA)[curr_ac]  # one-hot action
            
            # move in direction and get environment output
            s, r, terminal, _ = env.step(curr_ac.cpu().detach().numpy())
            # add to history
            states.append(np.expand_dims(s, axis=0))
            actions.append(action_OH)
            action_logprobs.append(ac_logprob)
            rewards.append(r)
            curr_state = copy.deepcopy(s)
        if DEBUG: print(f"action probs:{ac_logprob}\nfinal action:{action_OH}")
        print("Finished after {} timesteps".format(cts+1))
		# flatten 
        return np.stack(states), np.stack(actions), np.stack(rewards), torch.stack(action_logprobs)

    def get_G(self, rewards, gamma, a2c_v_end=None):
        # get sum of discounted reward vector : [sum : r*gamma*0...r*gamma*n]
        G_t=[0]
        if self.type == 0 or self.type == 1:
            for n_gamma, r in enumerate(reversed(rewards)):
                # insert from last reward to beginning 
                G_t.insert(0, r+gamma**n_gamma*G_t[0])
            if DEBUG: print(f"Sum of Expected Rewards G:{G_t}")
            G_t=np.array(G_t[:-1])
        elif self.type == 2:
            a2c_v_end = a2c_v_end.cpu().detach().numpy().flatten()
            for t in range(len(rewards)):
                g_t = 0 if t+self.n>len(rewards) else (gamma**self.n)*a2c_v_end[t+self.N]
                for k in range(min(self.N-1, len(rewards)-1)):
                    g_t += (gamma**k)*rewards[t+k]
                G_t.append(g_t)
        return torch.FloatTensor(G_t).to(device)
        
    def train(self, env, gamma=0.99, n=10):
        """
        # Trains the model on a single episode using REINFORCE or A2C/A3C.
        params:
            n: number of n-steps look ahead
        """
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        eps_lim = 500    # length of episode 'limit'
        batch_size_ratio = 1    # portion of length of states to feed in NN
        assert 0<batch_size_ratio<=1, "Batch size between 0 and 1"
        #
        env = wrap_env(env)
        mean_total, std_total = [], []
        # generate episode
        states, actions, rewards, action_logprobs = self.generate_episode(env)
        states = torch.Tensor(states[:-1]).to(device)   # remove the last pseudo state
        if self.type != 2:   # Reinforce / baseline
            G_t = self.get_G(rewards, gamma)
        if self.type == 1 or self.type == 2:  # baseline/A2C subtraction
            baseline_value = self.critic(states)   # (1, t)
            if self.type ==2:  # A2C N-step 
                G_t = self.get_G(rewards, gamma, baseline_value)
            G_t_adj = torch.subtract(G_t, baseline_value)
        else:
            G_t_adj = G_t  # no baseline
        # weight actions by rewards/advantage and fit model 
        actor_history = self.fit_model(
            action_logprobs, G_t_adj, self.actor_optimizer, self.reinforce_criterion, batch_size=int(len(states)*batch_size_ratio),
        )
        if self.type == 1 or self.type == 2:   # require a baseline value (A2C/Baseline)
            critic_history = self.fit_model(
                baseline_value, G_t, self.critic_optimizer, self.baseline_criterion, batch_size=int(len(states)*batch_size_ratio),
            )
        return actor_history
    
def main_a2c(args):
    # Parse command-line arguments.
    env_name = args.env_name

    # Create the environment.
    env =  wrap_env(gym.make(env_name))
    nA = env.action_space.n
    nS = env.observation_space.shape[0]
    print(f"Configurations:{args}")

    # Plot average performance of 5 trials
    num_seeds = 5   # 5
    eval_freq = 100  # 100
    l = args.num_episodes//eval_freq
    res = np.zeros((num_seeds, l))

    gamma = 0.99
    for i in tqdm.tqdm(range(num_seeds)):
        print(f"Seed:{i}")
        # Fix seed
        torch.manual_seed(i)
        np.random.seed(i)
        # save mean evaluation reward
        reward_means = []
        # TODO: create networks and setup reinforce/a2c
        history = dict.fromkeys(['train','test'],[])
        actor = NeuralNet(input_size=nS, output_size=nA, 
                          activation=nn.Softmax(dim=1)).to(device)
        critic = NeuralNet(input_size=nS, output_size=1, 
                           activation=nn.LeakyReLU(0.9)).to(device)
        A2C_net = A2C(actor=actor, actor_lr=args.lr, N=args.n, nA=nA, 
                      critic=critic, critic_lr=args.critic_lr, baseline=args.use_baseline, a2c=False)
        for m in range(args.num_episodes):
            print("Episode: {}".format(m))
            history['train'].append(A2C_net.train(env, gamma=gamma))
            if m % eval_freq == 0:
                print("[Policy Evaluation]")
                G = np.zeros(20)   # save 20 iterations of evaluation 
                for k in range(20):
                    g = A2C_net.evaluate_policy(env)
                    G[k] = g
                reward_mean = G.mean()
                reward_sd = G.std()
                print("The test reward for episode {0} is {1} with sd of {2}.".format(m, reward_mean, reward_sd))
                reward_means.append(reward_mean)
                history['test'].append(G)
        res[i] = np.array(reward_means)
    return history, res
    
if __name__=="__main__":
    import argparse, matplotlib.pyplot as plt, tqdm
    def parse_a2c_arguments():
        # Command-line flags are defined here.
        parser = argparse.ArgumentParser()
        parser.add_argument('--env-name', dest='env_name', type=str,
                            default='CartPole-v0', help="Name of the environment to be run.")   # 'LunarLander-v2'
        parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                            default=10, help="Number of episodes to train on.")    # 3500
        parser.add_argument('--lr', dest='lr', type=float,
                            default=5e-4, help="The actor's learning rate.")
        parser.add_argument('--use_baseline', dest='use_baseline', type=bool,
                            default=False, help="Use baseline model")
        parser.add_argument('--baseline-lr', dest='baseline_lr', type=float,
                            default=5e-4, help="The actor's learning rate.")
        parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                            default=1e-4, help="The critic's learning rate.")
        parser.add_argument('--n', dest='n', type=int,
                            default=100, help="The value of N in N-step A2C.")

        parser_group = parser.add_mutually_exclusive_group(required=False)
        parser_group.add_argument('--render', dest='render',
                                  action='store_true',
                                  help="Whether to render the environment.")
        parser_group.add_argument('--no-render', dest='render',
                                  action='store_false',
                                  help="Whether to render the environment.")
        parser.set_defaults(render=False)

        return parser.parse_args()
    
    args = parse_a2c_arguments()
    history, res = main_a2c(args)
    
    ks = np.arange(l)*100
    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    plt.fill_between(ks, mins, maxs, alpha=0.1)
    plt.plot(ks, avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    if not os.path.exists('./plots'):
        os.mkdir('./plots')

    if A2C_net.type == 'A2C' or A2C_net.type == 2:
        plt.title("A2C Learning Curve for N = {}".format(args.n), fontsize = 24)
        plt.savefig("./plots/a2c_curve_N={}.png".format(args.n))
    elif A2C_net.type == 'Baseline' or A2C_net.type == 1:
        plt.title("Baseline Reinforce Learning Curve".format(args.n), fontsize = 24)
        plt.savefig("./plots/Baseline_Reinforce_curve.png".format(args.n))
    elif A2C_net.type == 'Reinforce' or A2C_net.type == 0: # Reinforce
        plt.title("Reinforce Learning Curve", fontsize = 24)
        plt.savefig("./plots/Reinforce_curve.png")
    plt.plot(res)