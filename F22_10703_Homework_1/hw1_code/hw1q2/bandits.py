import numpy as np
import matplotlib.pyplot as plt
import math
import random 


## PROBLEM 2 : BANDITS
## In this section, we have given you a template for coding each of the 
## exploration algorithms: epsilon-greedy, optimistic initialization, UCB exploration, 
## and Boltzmann Exploration 

## You will be implementing these algorithms as described in the “10-armed Testbed” in Sutton+Barto Section 2.3
## Please refer to the textbook or office hours if you have any confusion.

## note: you are free to change the template as you like, do not think of this as a strict guideline
## as long as the algorithm is implemented correctly and the reward plots are correct, you will receive full credit

# This is the optional wrapper for exploration algorithm we have provided to get you started
# this returns the expected rewards after running an exploration algorithm in the K-Armed Bandits problem
# we have already specified a number of parameters specific to the 10-armed testbed for guidance
# iterations is the number of times you run your algorithm

# WRAPPER FUNCTION
def explorationAlgorithm(explorationAlgorithm, param, iters):
    cumulativeRewards = []
    for i in range(iters):
        # number of time steps
        t = 1000
        # number of arms, 10 in this instance
        k = 10
        # real reward distribution across K arms
        rewards = np.random.normal(1, 1, k)
        # counts for each arm
        n = np.zeros(k)
        # extract expected rewards by running specified exploration algorithm with the parameters above
        # param is the different, specific parameter for each exploration algorithm
        # this would be epsilon for epsilon greedy, initial values for optimistic intialization, c for UCB, and temperature for Boltmann 
        currentRewards = explorationAlgorithm(param, t, k, rewards, n)
        cumulativeRewards.append(currentRewards)
    # TO DO: CALCULATE AVERAGE REWARDS ACROSS EACH ITERATION TO PRODUCE EXPECTED REWARDS
    expectedRewards = np.vstack(cumulativeRewards).mean(axis=0)
    return expectedRewards

# EPSILON GREEDY TEMPLATE
def epsilonGreedy(epsilon, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    rewards_per_step = []
    # TO DO: initialize an initial q value for each arm
    Q_est = np.zeros(k)
    # TO DO: implement the epsilon-greedy algorithm over all steps and return the expected rewards across all steps
    
    for _ in range(steps):
        p = random.uniform(0,1)
        if p<=1-epsilon:
            greedy_actions = np.where(Q_est == np.max(Q_est))[0]
            action = np.random.choice(greedy_actions)
        else: 
            action = np.random.choice(k)

        reward = realRewards[action] + np.random.normal(0,1)

        n[action] += 1
        Q_est[action] += ((reward - Q_est[action]) / n[action])

        greedy_actions = np.where(Q_est == np.max(Q_est))[0]

        expected_reward = (1-epsilon)*realRewards[greedy_actions].mean() + epsilon*realRewards.mean()
        rewards_per_step.append( expected_reward )

    return rewards_per_step

# OPTIMISTIC INTIALIZATION TEMPLATE
def optimisticInitialization(value, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    
    # TO DO: initialize optimistic initial q values per arm specified by parameter
    
    # TO DO: implement the optimistic initializaiton algorithm over all steps and return the expected rewards across all steps
    rewards_per_step = []

    Q_est = np.zeros(k) + value

    for _ in range(steps):

        greedy_actions = np.where(Q_est == np.max(Q_est))[0]
        action = np.random.choice(greedy_actions)

        reward = realRewards[action] + np.random.normal(0,1)

        n[action] += 1
        Q_est[action] += ((reward - Q_est[action]) / n[action])

        greedy_actions = np.where(Q_est == np.max(Q_est))[0]

        expected_reward = realRewards[greedy_actions].mean() 
        rewards_per_step.append( expected_reward )

    return rewards_per_step

# UCB EXPLORATION TEMPLATE
def ucbExploration(c, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step

    # TO DO: initialize q values per arm 

    # TO DO: implement the UCB exploration algorithm over all steps and return the expected rewards across all steps
    rewards_per_step = []

    Q_est = np.zeros(k)

    for t in range(1, steps+1):

        upper_bounds = Q_est + c* np.sqrt(np.log(t) / (n+1e-6) )

        greedy_actions = np.where(upper_bounds == np.max(upper_bounds))[0]
        action = np.random.choice(greedy_actions)

        reward = realRewards[action] + np.random.normal(0,1)

        n[action] += 1
        Q_est[action] += ((reward - Q_est[action]) / n[action])

        upper_bounds = Q_est + c* np.sqrt(np.log(t) / (n+1e-6) )
        greedy_actions = np.where(upper_bounds == np.max(upper_bounds))[0]

        expected_reward = realRewards[greedy_actions].mean() 
        rewards_per_step.append( expected_reward )

    return rewards_per_step


# BOLTZMANN EXPLORATION TEMPLATE
def boltzmannE(temperature, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step

    # TO DO: initialize q values per arm 

    # TO DO: initialize probability values for each arm

    # TO DO: implement the Boltzmann Exploration algorithm over all steps and return the expected rewards across all steps
    rewards_per_step = []

    Q_est = np.zeros(k)

    for _ in range(steps):

        pi = temperature * Q_est
        pi -= pi.max() # to avoid overflow in exponential 
        pi = np.exp(pi) 
        pi /= pi.sum()

        # select action based on boltzmann energy probability
        action = np.random.choice(k, p=pi)

        reward = realRewards[action] + np.random.normal(0,1)

        # update boltzmann action distribution for the particular ation
        n[action] += 1
        Q_est[action] += ((reward - Q_est[action]) / n[action])

        # compute expected reward
        # recompute distribution for expectation 
        pi = temperature * Q_est
        pi -= pi.max() # to avoid overflow in exponential 
        pi = np.exp(pi) 
        pi /= pi.sum()

        expected_reward = (pi*realRewards).sum() 
        rewards_per_step.append( expected_reward )

    return rewards_per_step

# PLOT TEMPLATE
def plotExplorations(paramList, algo):
    # TO DO: for each parameter in the param list, plot the returns from the exploration Algorithm from each param on the same plot
    x = np.arange(1,1001)
    # calculate your Ys (expected rewards) per each parameter value
    # plot all the Ys on the same plot
    # include correct labels on your plot!
    iters=20
    plt.figure()
    for param in paramList:
        expected_rewards = explorationAlgorithm(algo, param, iters)
        plt.plot(x, expected_rewards, label=param)
    plt.title("Mean Expected Reward vs. Timestep, {}".format(algo.__name__))
    plt.xlabel("t")
    plt.ylabel("Expected Reward")

    plt.legend()
    plt.show()

if __name__=="__main__":
    # # 2.1 epsilon greedy
    # plotExplorations([0, 0.001, 0.01, 0.1, 1.0], epsilonGreedy)
    # # 2.2 optimistic
    # plotExplorations([0, 1, 2, 5, 10], optimisticInitialization)
    # # 2.3 ucb
    # plotExplorations([0, 1, 2, 5], ucbExploration)
    # # 2.4 boltzmann 
    # plotExplorations([1, 3, 10, 30, 100], boltzmannE)

    x = np.arange(1,1001)
    iters=20
    plt.figure()

    for algo, param in [(epsilonGreedy, 0.1), (optimisticInitialization, 5), (ucbExploration, 5), (boltzmannE, 3)]:
        expected_rewards = explorationAlgorithm(algo, param, iters)
        plt.plot(x, expected_rewards, label=algo.__name__ + ", " + str(param), alpha=0.7)

    plt.title("Best Performing Hyperparams: Mean Expected Reward vs Timestep")
    plt.xlabel("t")
    plt.ylabel("Expected Reward")
    plt.legend()

    plt.savefig("2-5.png")

    plt.show()
