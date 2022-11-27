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
        self.max_logvar = torch.tensor(-3 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)
        self.min_logvar = torch.tensor(-7 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)

        # Create or load networks
        self.networks = nn.ModuleList([self.create_network(n) for n in range(self.num_nets)]).to(device=self.device)
        self.opt = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)


        # EDIT : init a gaussian log likelihood loss module
        self.gaussian_nlll = nn.GaussianNLLLoss()

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float)
        return [self.get_output(self.networks[i](inputs)) for i in range(self.num_nets)]

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

    def get_loss(self, targ, mean, logvar):
        # TODO: write your code here
        '''
        UNTESTED, PROBABLY HAS ERRORS
        '''
        return self.gaussian_nlll(input=mean, target=targ, var=torch.exp(logvar))

    def create_network(self, n):
        layer_sizes = [self.state_dim + self.action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, HIDDEN3_UNITS]
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()]
                         for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2 * self.state_dim)]
        return nn.Sequential(*layers)

    def train_model(self, inputs, targets, batch_size=128, num_train_itrs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 2)
        Argument:
          inputs: state and action inputs. Assumes that inputs are standardized.
          targets: ----resulting states---- EDIT: from Piazza, this should be delta from one state to the next; delta targets
        Return:
            List containing the average loss of all the networks at each train iteration

        """
        # TODO: write your code here

        '''
        UNTESTED, PROBABLY HAS ERRORS
        '''

        losses = [[] for _ in range(self.num_nets)]

        n = len(targets)
        for _ in range(num_train_itrs):
            output_list = self.forward(inputs) # list of each network's raw outputs
            for net_idx, output in enumerate(output_list): # iterate for each network's output
                
                # sample a minibatch
                batch_idx = np.random.choice(n, size=batch_size)
                batch_inputs = torch.tensor( inputs[batch_idx] , device=self.device, dtype=torch.float)
                batch_targets = torch.tensor( targets[batch_idx] , device=self.device, dtype=torch.float)

                self.opt.zero_grad()
                mean, logvar = self.get_output(self.networks[net_idx](batch_inputs) )   
                loss = self.get_loss(batch_targets, mean, logvar)
                loss.backward()
                self.opt.step()
                
                losses[net_idx].append(loss.detach().cpu())
        losses = np.mean( np.array(losses), axis=0 )
        return losses
