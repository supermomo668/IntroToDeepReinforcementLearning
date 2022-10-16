import sys
import argparse
import numpy as np

import sys, os, copy
from pathlib import Path

import numpy as np, torch, gym
import torch.nn.functional as F
from gym.wrappers import record_video


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
    env = record_video(env, proj_folder/'video', name_prefix='train')
    return env
    
class A2C(object):
    # Implementation of N-step Advantage Actor Critic.

    def __init__(self, actor, actor_lr, N, nA, critic, critic_lr, baseline=False, a2c=True):
        # Note: baseline is true if we use reinforce with baseline
        #       a2c is true if we use a2c else reinforce
        
        # TODO: Initializes A2C.
        self.type = "A2C" if a2c else ("Baseline" if baseline else "Reinforce")  # Pick one of: "A2C", "Baseline", "Reinforce"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor.to(device)
        # define meta variabless
        self.save_dir = Path('./').absolute()/'Output'
        if self.type == "A2C":
            critic.to(device)
            self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        elif self.type == "Baseline":
            pass
        else:
            # Reinforce
            pass
        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.N = N
        assert self.type is not None, "Type must be provided"

    def reinforce_criterion(self, y_pred, y_true):
        log_y_pred = torch.log(y_pred)   # log probability of ation
        l = torch.multiply(log_y_pred, y_true)   # (n_eps, nA)
        l_action = torch.sum(l, axis=1)  # (n_eps,)
        return torch.mean(l_action)
    
    def fit_actor(self, states, G_total, epochs=1, batch_size=1):
        """
        """
        self.actor.train()
        n_example = len(states)
        history = dict.fromkeys(['loss'],[])
        for e in range(epochs):
            for i in range(int(np.ceil(len(states)/batch_size))):
                start_idx, end_idx = batch_size*i, min(batch_size*(i+1), n_example)
                states_, target_ = states[start_idx: end_idx], G_total[start_idx: end_idx]
                pred_G = self.actor(states_.cuda())
                loss = self.reinforce_criterion(pred_G, target_.cuda())
                # # measure metrics and record loss
                # m1 = metric1(cls_out.cpu(), target)
                # m2 = metric2(cls_out.cpu(), target)
                history['loss'].append(loss)
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                if DEBUG: print(f"Latest Loss:{history['loss'][-1]}")
        return history

    def evaluate_policy(self, env):
        # TODO: Compute Accumulative trajectory reward(set a trajectory length threshold if you want)
        """ compute rewards
        """
        self.actor.eval()
        _, _, r, _ = self.generate_episode(env, render=False)
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
        states, actions, rewards, actions_probs=[], [] ,[], []
        # 
        nS = env.observation_space.shape[0]
        nA = env.action_space.n
		# Start episode
        states.append(np.expand_dims(env.reset(), axis=0))
        terminal = False
        if render: env.render(mode='human')
        cts = 0
        while not terminal:
            cts+=1
            ac = self.actor(torch.Tensor(states[-1]).to(device)).squeeze(0)   # ensure [nA] vector
            if DEBUG: print(f"Input state:{states[-1]}\nOutput action:{ac}")
            ac_prob = ac.detach().cpu().numpy().flatten()
            ac_prob = np.nan_to_num(ac_prob,0)
            #
            a_ = np.random.choice(ac_prob, p=ac_prob)   # stochastic choice
            curr_ac = np.where(ac_prob==a_)[0][0]  # Current Action
            action_OH = np.eye(nA)[curr_ac]  # one-hot action
            
            # move in direction and get environment output
            s, r, terminal, _ = env.step(curr_ac)
            # add to history
            states.append(np.expand_dims(s, axis=0))
            actions.append(action_OH)
            actions_probs.append(ac_prob)
            rewards.append(r)
            curr_state = copy.deepcopy(s)
        if DEBUG: print(f"action probs:{ac_prob}\nfinal action:{action_OH}")
        print("Finished after {} timesteps".format(cts+1))
		# flatten 
        states=np.reshape(np.array(states), (-1, nS))
        actions=np.reshape(np.array(actions), (-1, nA))
        return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(actions_probs)

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
        alpha = 1e-2
        assert 0<batch_size_ratio<=1, "Batch size between 0 and 1"
        #
        env = wrap_env(env)
        mean_total, std_total = [], []
        # generate episode
        states, actions, rewards, actions_prob = self.generate_episode(env, render=True)
        states = states[:-1]   # remove the last state
        # get discounted reward vector
            # [sum : r*gamma*0...r*gamma*n]
        G_tot=[0]
        for n_gamma, r in enumerate(reversed(rewards)):
            # insert from last reward to beginning 
            G_tot.insert(0, r+gamma**n_gamma*G_tot[0])
        if DEBUG: print(f"Sum of Expected Rewards G:{G_tot}")
        G_tot=np.array(G_tot[:-1])
        # weight actions by rewards and fit model
        if DEBUG: print(f"G_tot:[{len(G_tot)}], actions:[{len(actions)}], states:[{len(states)}]")
        G_total_actions = np.multiply(G_tot, actions.T).T   # (t, nA)
        train_history = self.fit_actor(torch.Tensor(states), torch.Tensor(G_total_actions*alpha), epochs=1, batch_size=int(len(states)*batch_size_ratio))
        return train_history