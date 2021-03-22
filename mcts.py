import numpy as np
from collections import defaultdict
from math import copysign

class MCTS:
    """
    An implementation of the Monte Carlo Tree Search algorithm. Agents trained using this algorithm learn to
    solve an environment by accumulating action value estimates backedup through Monte Carlo simulations.
    
    Attributes:
        env: The openAI gym environment being learned by the MCTS agent.
        c: The exploration coefficient c for the UCB policy.
        cheating: A binary value which determines if the environment's reward policy should be
        modified (assuming FrozenLake).
    """
    
    def __init__(self, env, c=1, cheating=False):
        self.t = defaultdict(lambda: 0)
        self.Q = defaultdict(lambda: np.zeros(env.nA))
        self.N = defaultdict(lambda: np.zeros(env.nA))
        self.env = env
        self.c = c
        self.trajectory = []
        self.cheating = cheating
     
    def divide(self, numerator, denominator):
        """
        Courtesy of: https://stackoverflow.com/a/16226757
        Performs division operation while allowing for results of inf/-inf when
        dividing by 0/-0.

        Args:
            numerator: The value of the numerator.
            denominator: The value of the denominator.
        Returns:
            The result of dividing the numerator by the denominator. 
        """
        if denominator == 0.0:
            return copysign(float('inf'), denominator)
        return numerator / denominator
        
    def UBC_policy(self, node, action):
        """
        Calculates the UCB score for a given state-action pair.
        
        Args:
            node: The current node(state) of the traversal.
            action: The current action being considered for seletion.
        Returns:
            new_estimate: The UCB score for this state-action pair.
        """
        action_value = self.divide(self.Q[node][action],self.N[node][action])
        with np.errstate(divide='ignore'):
            new_estimate = action_value + self.c*np.sqrt(self.divide(np.log(self.t[node]),self.N[node][action]))
        return new_estimate
    
    def add_state_action_pair(self, node, action):
        """
        Updates the counts for the total number of visits to a node(state), and the number of times an action
        from that node has been taken. Both the node and action are added to the current traversal's
        trajectory for backup.
        
        Args:
            node: The current node(state) of the traversal.
            action: The selected action for traversal.
        """
        self.trajectory.append((node,action))
        self.t[node] += 1
        self.N[node][action] += 1
    
    def selection(self, node):
        """
        Selects actions based on the UCB score for each state-action pair.
        
        Args:
            node: The current node(state) of the traversal.
        Returns:
            The action with the largest UCB score from the current node.
        """
        
        UCBs = []
        for action in range(self.env.nA):
            UCBs.append(self.UBC_policy(node, action))
        return np.argmax(UCBs)
        
    def simulation(self, starting_node, max_t=10000):
        """
        The agent behaves using a stochastic rollout policy until the simulation is halted.
        
        Args:
            starting_node: The starting node(state) of the simulation.
            max_t: The maximum number of steps the simulation is allowed to run.
        Returns:
            total_reward: The total reward the agent recieved during the simulation
        """
        
        action = self.env.action_space.sample()
        self.add_state_action_pair(starting_node, action)
        total_reward = 0
        for t in range(max_t):
            next_state, reward, done, info = self.env.step(action)
            if self.cheating:
                total_reward += reward - 0.01
            else:
                total_reward += reward            
            if done:
                break
            else:        
                state = next_state
                action = self.env.action_space.sample()
        return total_reward
    
    def backup(self, total_reward):
        """
        The total reward gathered from a previous simulation or traversal of the tree is backedup to each of the visited
        nodes (states) in the tree.
        
        Args:
            total_reward: The reward to be backedup through all of the state-action pairs present in the current
            traversal's trajectory.
        """
        for i in range(len(self.trajectory)):
            state = self.trajectory[i][0]
            action = self.trajectory[i][1]
            self.Q[state][action] += total_reward
            
    def traverse(self, render=False, cheating=False):
        """
        Performs one iteration (episode) of a Monte Carlo Tree Search. Each iteration will help accumulate 
        more accurate action-value estimates.
        
        Args:
            render: Whether or not each of the agent's steps in the environment should be rendered.
            cheating: A binaray value which modifies the reward function of the environment (assuming FrozenLake).
        Returns:
            total_reward: The total reward gathered by the agent for one episode.
        """
        
        self.trajectory = []
        current_node = self.env.reset()
        action = self.selection(current_node)
        self.add_state_action_pair(current_node, action)
        if render:
             self.env.render()
        total_reward = 0
        done = False
        while self.N[current_node][action] > 1:
            current_node, reward, done, _ = self.env.step(action)
            if self.cheating:
                reward *= 10
                total_reward += reward-0.01
            else:
                total_reward += reward
            
            if done:
                self.backup(total_reward)
                break
            else:
                action = self.selection(current_node)
                self.add_state_action_pair(current_node, action)
                if render:
                    self.env.render()
        if not done:
            current_node, _, _, _ = self.env.step(action)
            total_reward = self.simulation(current_node)
            self.backup(total_reward)
        return total_reward