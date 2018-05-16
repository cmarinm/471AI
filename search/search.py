# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
from collections import namedtuple

import util
import searchAgents



class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

Node = namedtuple('Node', 'state action parent cost')       # we keep the parent in the node to track back actions,
                                                            # simpler than keeping list of actions on every node, space efficient too

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    currstate = problem.getStartState()                     # start state

    fringe = util.Stack()                                   # start fringe
    final_actions = []                                      # what we will return
    explored = dict()                                       # a set of explored states
    goalFound = False                                       # bool for when goal is found

    currnode = Node(currstate, None, None, 0)
    fringe.push(currnode)

    while not goalFound and not fringe.isEmpty():           # check if fringe is empty in so we don't pop an empty
        currnode = fringe.pop()
        currstate = currnode.state
        h = hash(currstate)                                 # set was giving trouble with state being more than a single value
                                                            # so I used a dict, easier for state being more than a single thing
        if explored.has_key(h):                             # ignore explored states
            continue

        explored[h] = 1                                     # mark node as visited
        if problem.isGoalState(currstate):                  # check if goal state, goalFound used to avoid 'break'
            goalFound = True
            continue

        successors = problem.getSuccessors(currstate)       # get successors

        for x in successors:
            if not explored.has_key(hash(x[0])):            # don't add explored states
                children = Node(x[0], x[1], currnode, x[2])
                fringe.push(children)



    while currnode.parent != None:                          # backtrack parents and actions to get final list of actions needed
        final_actions.insert(0, currnode.action)
        currnode = currnode.parent

    return final_actions



    # util.raiseNotDefined()




def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    ## Exact Same Code, but fringe is a queue, instead of a stack ##
    currstate = problem.getStartState()  # start state

    fringe = util.Queue()  # start fringe
    final_actions = []  # what we will return
    explored = dict()  # a set of explored states
    goalFound = False  # bool for when goal is found

    currnode = Node(currstate, None, None, 0)
    fringe.push(currnode)

    while not goalFound and not fringe.isEmpty():
        currnode = fringe.pop()
        currstate = currnode.state
        h = hash(currstate)
        if explored.has_key(h):
            continue

        explored[h] = 1
        if problem.isGoalState(currstate):
            goalFound = True
            continue

        successors = problem.getSuccessors(currstate)

        for x in successors:
            if not explored.has_key(hash(x[0])):
                children = Node(x[0], x[1], currnode, x[2])
                fringe.push(children)

    while currnode.parent != None:
        final_actions.insert(0, currnode.action)
        currnode = currnode.parent

    return final_actions




    # util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    ## Almost same code, but fringe is a PriorityQueue, instead of a stack ##
    currstate = problem.getStartState()  # start state

    fringe = util.PriorityQueue()  # start fringe
    final_actions = []  # what we will return
    explored = dict()  # a set of explored states
    goalFound = False  # bool for when goal is found

    currnode = Node(currstate, None, None, 0)
    fringe.push(currnode, currnode.cost)

    while not goalFound and not fringe.isEmpty():
        currnode = fringe.pop()
        currstate = currnode.state
        pcost = currnode.cost                                    # Keep track of parent's cost
        h = hash(currstate)
        if explored.has_key(h):
            continue

        explored[h] = 1
        if problem.isGoalState(currstate):
            goalFound = True
            continue

        successors = problem.getSuccessors(currstate)

        for x in successors:
            if not explored.has_key(hash(x[0])):
                children = Node(x[0], x[1], currnode, x[2] + pcost)
                fringe.push(children, children.cost)    # add parent's cost

    while currnode.parent != None:
        final_actions.insert(0, currnode.action)
        currnode = currnode.parent

    return final_actions


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    ## Almost same code, but fringe is a PriorityQueue, instead of a stack ##
    currstate = problem.getStartState()  # start state

    fringe = util.PriorityQueue()  # start fringe
    final_actions = []  # what we will return
    explored = dict()  # a set of explored states
    goalFound = False  # bool for when goal is found

    currnode = Node(currstate, None, None, 0)
    heur = heuristic(currstate, problem)
    fringe.push(currnode, currnode.cost + heur)

    while not goalFound and not fringe.isEmpty():
        currnode = fringe.pop()
        currstate = currnode.state
        pcost = currnode.cost  # Keep track of parent's cost
        h = hash(currstate)
        if explored.has_key(h):
            continue

        explored[h] = 1
        if problem.isGoalState(currstate):
            goalFound = True
            continue

        successors = problem.getSuccessors(currstate)

        for x in successors:
            if not explored.has_key(hash(x[0])):
                children = Node(x[0], x[1], currnode, x[2] + pcost)
                heur = heuristic(children.state, problem)
                fringe.push(children, children.cost + heur)  # add parent's cost

    while currnode.parent != None:
        final_actions.insert(0, currnode.action)
        currnode = currnode.parent

    return final_actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
