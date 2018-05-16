# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        legalMoves.remove("Stop")
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best


        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        capsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]



        "*** YOUR CODE HERE ***"
        scared = False
        for time in newScaredTimes:
            if time > 2: scared = True

        for ghost in newGhostStates:
            ghostp = ghost.getPosition()
            gdis = util.manhattanDistance(newPos,ghostp)
            if(scared):
                if (gdis <= 2): return 10
                if (gdis <= 1): return 20
            else:
                if(gdis <= 3): return 0
                if(gdis <= 2): return -1




        minsf = 9999
        foodList = newFood.asList()

        if(currentGameState.hasFood(newPos[0], newPos[1])): return 1000
        for fpos in foodList:
            fdis = util.manhattanDistance(newPos,fpos)
            if(fdis < minsf): minsf = fdis
            if minsf <= 1: break

        if minsf == 0: minsf = 0.1

        foodev = (10/minsf)



        return foodev

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    minmaxdepth = 0
    agents = 0
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        self.agents = gameState.getNumAgents()

        self.minmaxdepth = (self.depth*self.agents)-1 #get the 'real' length of the tree.
        actions = gameState.getLegalActions(0)        #start of minimax algorithm

        values = [self.minimaxValue(gameState.generateSuccessor(0, action), 1,1) for action in actions] # root node
        # eval this starts the recursive call to minimaxValue, at end of possible actions at root, select the max and
        #  return that action
        bestValue = max(values)
        chosenIndex=0
        for index in range(len(values)):
            if values[index] == bestValue: chosenIndex = index

        return actions[chosenIndex]

    def minimaxValue(self, gameState, currentDepth,ghost):
        tmpdepth = currentDepth+1                           #keep track of depth
        if(currentDepth > self.minmaxdepth): return self.evaluationFunction(gameState) #terminality, depth reached, then return
        else:
            if(ghost >= gameState.getNumAgents()): # max turn, this is when all min agent moves have been done (ghost
                #  count is for that)
                actions = gameState.getLegalActions(0)
                bestValue = -9999 #start at -inf
                terminalState = True # asume this is a terminal state (leaf) this could also be a terminality condition
                for action in actions: # leaf node if it has no children
                    successor = gameState.generateSuccessor(0,action)
                    if(successor != None):
                        terminalState = False #if it has children, then not terminalState
                        bestValue = max(bestValue,self.minimaxValue(successor, tmpdepth,1)) #best value is max

                if(terminalState): return self.evaluationFunction(gameState) #if it is a leaf node, return evaluation
                else: return bestValue # else, return the best value(through the recusive call to children node


            else:                     # min turn(s)
                bestValue = 9999
                actions = gameState.getLegalActions(ghost)
                terminalState = True #same here
                for action in actions:
                    successor = gameState.generateSuccessor(ghost,action)
                    if (successor != None):
                        terminalState = False
                        bestValue = min(bestValue, self.minimaxValue(successor, tmpdepth,ghost+1)) #best value is min
                if(terminalState): return self.evaluationFunction(gameState)
                else: return bestValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    minmaxdepth = 0
    agents = 0
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        " Almost exaclty the same as minimax, but now we add alpha and beta to every recursive call to minimaxValue"
        " And we use this to see if we can prune before going through the rest of the tree(we check at end of action" \
        "loop, before looping again to next action"
        self.agents = gameState.getNumAgents()

        self.minmaxdepth = (self.depth * self.agents) - 1
        actions = gameState.getLegalActions(0)
        alpha = -9999
        beta = 9999
        values = []
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            values.append(self.minimaxValue(successor,1,1,alpha,beta))
            bestValue = max(values)
            if bestValue > beta: return bestValue # this was causing trouble, also have to check for pruning at root
            alpha = max(alpha, bestValue)

        bestValue = max(values)
        chosenIndex = 0
        for index in range(len(values)):
            if values[index] == bestValue: chosenIndex = index


        return actions[chosenIndex]


    def minimaxValue(self, gameState, currentDepth, ghost, alpha, beta):
            tmpdepth = currentDepth + 1
            if (currentDepth > self.minmaxdepth):
                return self.evaluationFunction(gameState)  # terminality, depth
            else:
                if (ghost >= gameState.getNumAgents()):  # max turn
                    actions = gameState.getLegalActions(0)
                    bestValue = -9999
                    terminalState = True
                    for action in actions:
                        successor = gameState.generateSuccessor(0, action)
                        if (successor != None):
                            terminalState = False
                            bestValue = max(bestValue, self.minimaxValue(successor, tmpdepth, 1,alpha,beta))
                            if bestValue > beta: return bestValue # check beta to see if we can prune here
                            alpha = max(alpha,bestValue)          # update alpha


                    if (terminalState):
                        return self.evaluationFunction(gameState)
                    else:
                        return bestValue


                else:  # min turn(s)
                    bestValue = 9999
                    actions = gameState.getLegalActions(ghost)
                    terminalState = True
                    for action in actions:
                        successor = gameState.generateSuccessor(ghost, action)
                        if (successor != None):
                            terminalState = False
                            bestValue = min(bestValue, self.minimaxValue(successor, tmpdepth, ghost + 1,alpha,beta))
                            if bestValue < alpha: return bestValue # now we check alpha to see if we can prune
                            beta = min(beta, bestValue)            # update beta

                    if (terminalState):
                        return self.evaluationFunction(gameState)
                    else:
                        return bestValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    minmaxdepth = 0
    agents = 0
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        " Exactly the same as minimax agent, with one change on the min turn (ghost turns)"
        self.agents = gameState.getNumAgents()

        self.minmaxdepth = (self.depth * self.agents) - 1
        actions = gameState.getLegalActions(0)

        values = [self.minimaxValue(gameState.generateSuccessor(0, action), 1, 1) for action in actions]
        bestValue = max(values)
        chosenIndex = 0
        for index in range(len(values)):
            if values[index] == bestValue: chosenIndex = index

        return actions[chosenIndex]

    def minimaxValue(self, gameState, currentDepth, ghost):
        tmpdepth = currentDepth + 1
        if (currentDepth > self.minmaxdepth):
            return self.evaluationFunction(gameState)  # terminality, depth
        else:
            if (ghost >= gameState.getNumAgents()):  # max turn, stays same
                actions = gameState.getLegalActions(0)
                bestValue = -9999
                terminalState = True
                for action in actions:
                    successor = gameState.generateSuccessor(0, action)
                    if (successor != None):
                        terminalState = False
                        bestValue = max(bestValue, self.minimaxValue(successor, tmpdepth, 1))

                if (terminalState):
                    return self.evaluationFunction(gameState)
                else:
                    return bestValue


            else:  # min turn(s)
                bestValue = 0
                actions = gameState.getLegalActions(ghost)
                terminalState = True
                prob = 0
                if(len(actions) >0):
                    prob = 1.0/len(actions) # the probability of any action is random, equaly random
                for action in actions:
                    successor = gameState.generateSuccessor(ghost, action)
                    if (successor != None):
                        terminalState = False
                        bestValue = bestValue + prob*self.minimaxValue(successor, tmpdepth, ghost + 1) # don't get
                        # min, we get the average of all actions
                if (terminalState):
                    return self.evaluationFunction(gameState)
                else:
                    return bestValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    scared = False                           # ghost evaluation, distance to ghost, becomes positive and large
                                             # if ghosts are scared, negative if they are too close.
    ghostev = 0.0
    for time in newScaredTimes:
        if time > 2: scared = True
    for ghost in newGhostStates:
        ghostp = ghost.getPosition()
        gdis = util.manhattanDistance(newPos, ghostp) # distance to closes ghost
        if (scared):
            if (gdis <= 3): ghostev = 10.0*gdis
            if (gdis <= 2): ghostev = 20.0*gdis
        else:
            if (gdis <= 3): ghostev = -10.0
            if (gdis <= 2): ghostev = -30.0
            ghostev = gdis


    minsf = 9999.0                          # food evaluation, distance to closes food, take reciprocal because closer
                                            # is better
    foodList = newFood.asList()
    foodev = 0.0
    if (currentGameState.hasFood(newPos[0], newPos[1])):
        return 9000
    else:
        for fpos in foodList:
            fdis = util.manhattanDistance(newPos, fpos)
            if (fdis < minsf): minsf = fdis
            if minsf == 0 :break
        if minsf == 0: minsf = 0.1
        foodev = (1 / minsf)

    minsf = 9999.0                          # capsule evaluation, same but with capsules
    capev = 0.0
    for cpos in capsules:
        cdis = util.manhattanDistance(newPos, cpos)
        if (cdis < minsf): minsf = cdis
        if minsf <= 1: break
    if minsf == 0: minsf = 0.1
    capev = (1 / minsf)
    foodcount = newFood.count() + 1        # count of food left
    total = 50*foodev +50*ghostev + 0.1*currentGameState.getScore() + 10*capev + 30*(1/foodcount) # added game score to total evaluationm

    return total


# Abbreviation
better = betterEvaluationFunction

