import os
import numpy as np
import torch
import torch.nn as nn
import operator
from functools import reduce
from utils.util import ZFilter

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

import logging
log = logging.getLogger('root')

class ensemble(nn.Module):
    def __init__(self, num_nets, state_dim, action_dim, hidden_units=[HIDDEN1_UNITS, HIDDEN2_UNITS, HIDDEN3_UNITS]):
        super(ensemble, self).__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        self.linears = nn.ModuleList([self.create_network(hidden_units) for n in range(num_nets)])
        # Log variance bounds
        self.register_buffer('max_logvar', torch.tensor(-3 * np.ones([1, self.state_dim]), dtype=torch.float))
        self.register_buffer('min_logvar', torch.tensor(-7 * np.ones([1, self.state_dim]), dtype=torch.float))
        
    def create_network(self, hidden_units):
        # Comment: takes input size (state_dim+action_dim) output (state_dim)
        layer_sizes = [self.state_dim + self.action_dim, *hidden_units]
        layers = reduce(operator.add, [[nn.Linear(a, b), nn.ReLU()]
                                       for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2 * self.state_dim)]
        return nn.Sequential(*layers)
    
    def get_output(self, output):
        """
        Argument:
          output: the raw output of a single ensemble member
        Return:
          mean and log variance
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar
    
    
    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        out=[]
        for m in self.linears:
            out.append(self.get_output(m(x)))
        return out

class PENN(nn.Module):
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device=None):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        super().__init__()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Log variance bounds
        # Create or load networks
        #self.networks = nn.ModuleList([self.create_network(n) for n in range(self.num_nets)]).to(device=self.device)
        self.networks = ensemble(self.num_nets, self.state_dim, self.action_dim).to(device=self.device)
        self.opt = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)
        # EDIT
        self.criterion = nn.GaussianNLLLoss()
        #self.device =  next(self.networks.parameters()).device

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float)
        return self.networks(inputs)
        #return [self.get_output(self.networks[i](inputs)) for i in range(self.num_nets)]

    # def get_output(self, output):
    #     """
    #     Argument:
    #       output: the raw output of a single ensemble member
    #     Return:
    #       mean and log variance
    #     """
    #     mean = output[:, 0:self.state_dim]
    #     raw_v = output[:, self.state_dim:]
    #     logvar = self.networks.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
    #     logvar = self.networks.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
    #     return mean, logvar

    def get_loss(self, targ, mean, logvar):
        # TODO: write your code here
        # EDIT
        return self.criterion(mean, targ, torch.exp(logvar))
        raise NotImplementedError

    def create_network(self, n):
        # Comment: takes input size (state_dim+action_dim) output (state_dim)
        layer_sizes = [self.state_dim + self.action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, HIDDEN3_UNITS]
        layers = reduce(operator.add, [[nn.Linear(a, b), nn.ReLU()]
                                       for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2 * self.state_dim)]
        return nn.Sequential(*layers)

    def train_model(self, inputs, targets, batch_size=128, num_train_itrs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 2)
        Argument:
          inputs: state and action inputs. Assumes that inputs are standardized.
          targets: resulting states
        Return:
            List containing the average loss of all the networks at each train iteration

        """
        # TODO: write your code here
        # EDIT
        losses = [[] for _ in range(len(range(num_train_itrs)))]
        assert inputs.shape[0]==targets.shape[0], f"Shapes are:{inputs.shape, targets.shape}"
        n = len(targets)
        for it in range(num_train_itrs):
            output_list = self.forward(inputs) # list of each network's raw outputs
            for idx_start in range(0, n, batch_size):
                batch_inputs = torch.tensor(inputs[idx_start: min(n, idx_start+batch_size)], dtype=torch.float, device=self.device)
                batch_targets = torch.tensor(targets[idx_start: min(n, idx_start+batch_size)], dtype=torch.float, device=self.device)
                # get output
                batch_out = self.forward(batch_inputs)    # n_nets * [(*, nS), (*, nS)]   # list
                batch_out = torch.stack(list(map(lambda x: torch.stack(x, axis=1), batch_out)), axis=1)   #  (*, n_nets, 2, nS)
                stacked_batched_inputs = torch.stack([batch_inputs[..., :self.state_dim]]*self.num_nets, axis=1) # (*, n_nets, nS)
                # optimize
                self.opt.zero_grad()
                loss = self.get_loss(stacked_batched_inputs, batch_out[...,0, :], batch_out[...,1, :])   # (*, n_nets, nS)
                loss.backward()
                self.opt.step()
                losses[it].append(loss.item())
        return np.array(losses).mean(axis=0)

        raise NotImplementedError