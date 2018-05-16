# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        nextIt = self.values.copy() #same as values but it is the next iteration (the next state vector k+1)

        for i in range(0,iterations):
            nextIt = self.values.copy()
            for state in states:
                tmpval = 0
                updateval = None # the final update value
                actions = self.mdp.getPossibleActions(state) #get all actions for that state
                qvals = []
                for action in actions:
                    tmpval = self.computeQValueFromValues(state,action) #calculate the Q value
                    qvals.append(tmpval)
                if(len(qvals) == 0):
                    updateval = 0
                else:
                    updateval = max(qvals) #get the max out of all q values

                nextIt[state] = updateval #update the k+1 vector with this value
            self.values = nextIt        #reiteration update





    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        nextStates = self.mdp.getTransitionStatesAndProbs(state,action) #list of next states as (NextState,prob)
        qval = 0
        for nextState in nextStates:
            reward = self.mdp.getReward(state,action,nextState[0]) # Get the reward R(s,a,s')
            prob = nextState[1] #the probability T(s,a,s')
            qval += prob*(reward + self.discount*self.values[nextState[0]]) # sum of T(s,a,s')[R(s,a,s') + disc*V(s')]
        return qval


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None

        val = -9999     # value = -inf
        chosenAction = None
        for action in actions:
            tmpval = self.computeQValueFromValues(state, action) # Compute Q value fot every action, get the highest
            if(tmpval > val):
                val = tmpval
                chosenAction = action

        return chosenAction # if there was no actions, it will return chosenAction which was initialized to None


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
