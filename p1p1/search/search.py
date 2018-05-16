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
    currxy = problem.getStartState()                        # start state

    fringe = util.Stack()                                   # start fringe
    actions = util.Stack()                                  # start action stack
    final_actions = []                                      # what we will return
    explored = set()                                        # a set of explored states
    goalFound = False                                       # bool for when goal is found

    if problem.isGoalState(currxy):                         # if start state is goal
        return []
    explored.add(currxy)                                    # add start to explored
    successors = problem.getSuccessors(currxy)              # get successors for root

    for x in successors:                                    # push into fringe
        fringe.push(x)

    while not goalFound:                                    # in order to keep actions stored, we do not remove
                                                            # explored node from the fringe until we come back to it
        currstate = fringe.pop()
        fringe.push(currstate)
        currxy = currstate[0]

        if currxy in explored:                              # when we come back to it, remove it
            fringe.pop()
            if not actions.isEmpty():                       # and then pop its action: its subtree does not contain goal
                actions.pop()
            continue

        explored.add(currxy)                                # add nodes to explored
        actions.push(currstate[1])                          # push its actions

        if problem.isGoalState(currxy):                     # found goal
            goalFound = True
            continue
        successors = problem.getSuccessors(currxy)          # if no successors, leaf node (not goal) pop its action
        if len(successors) == 0:
            actions.pop()

        for x in successors:                                # push all unexplored successors into fringe
            tmpxy = x[0]
            if tmpxy not in explored:
                fringe.push(x)

    while not actions.isEmpty():                            # put actions in a list, reverse it
        final_actions.append(actions.pop())
    final_actions.reverse()


    return final_actions


    # util.raiseNotDefined()


Node = namedtuple('Node', 'state action actionsSoFar cost') # from now on, having nodes with preceding actions on it
                                                            # is easier than to keep a global actions list


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    currxy = problem.getStartState()
    fringe = util.Queue()                                   # now fringe is a queue, not a stack
    explored = set()
    goalFound = False
    currnode = Node(currxy, 'West', [], 0)                  # simple initiate currnode dummy values

    if problem.isGoalState(currxy):                         # same procedure as DFS, now using the namedtuple instead
        return []

    explored.add(currxy)
    successors = problem.getSuccessors(currxy)

    for x in successors:
        newnode = Node(x[0], x[1], [], 0)                   # push successor nodes into fringe all costs 0: queue anyway
        newnode.actionsSoFar.append(x[1])
        fringe.push(newnode)

    while not goalFound:
        currnode = fringe.pop()
        currxy = currnode.state

        if currxy not in explored:                         # to avoid cycles, only check node if not already explored
            if problem.isGoalState(currxy):                # this was not necessary in DFS because of its search nature
                goalFound = True
                continue

            explored.add(currxy)
            successors = problem.getSuccessors(currxy)

            for x in successors:
                tmpxy = x[0]
                if tmpxy not in explored:                  # copy parent's actions into children then add their action
                    newnode = Node(x[0], x[1], list(currnode.actionsSoFar), 0)
                    newnode.actionsSoFar.append(x[1])
                    fringe.push(newnode)

    return currnode.actionsSoFar




    # util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    currxy = problem.getStartState()                        # pretty much the same as BFS
    fringe = util.PriorityQueue()                           # use a priority queue for fringe
    explored = set()
    goalFound = False
    currnode = Node(currxy, 'West', [], 0)

    if problem.isGoalState(currxy):
        return []

    explored.add(currxy)
    successors = problem.getSuccessors(currxy)

    for x in successors:
        newnode = Node(x[0], x[1], [], x[2])
        newnode.actionsSoFar.append(x[1])
        fringe.push(newnode, x[2])                          # all costs of first level are the original costs

    while not goalFound:
        currnode = fringe.pop()
        currxy = currnode.state

        if currxy not in explored:
            if problem.isGoalState(currxy):                 # check if state is goal
                goalFound = True
                continue

            explored.add(currxy)
            successors = problem.getSuccessors(currxy)

            for x in successors:
                tmpxy = x[0]
                if tmpxy not in explored:                   # all further nodes add cost of their parent to theirs
                    newnode = Node(x[0], x[1], list(currnode.actionsSoFar), x[2]+currnode.cost)
                    newnode.actionsSoFar.append(x[1])
                    fringe.push(newnode, newnode.cost)

    return currnode.actionsSoFar
   # util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    currxy = problem.getStartState()                        # pretty much the same as UCS
    fringe = util.PriorityQueue()                           # use a priority queue for fringe
    explored = set()
    goalFound = False
    currnode = Node(currxy, 'West', [], 0)
    heur = 0

    if problem.isGoalState(currxy):
        return []

    explored.add(currxy)
    successors = problem.getSuccessors(currxy)

    for x in successors:
        heur = heuristic(x[0], problem)                     # now we add the heuristic h(n) to g(n)
        newnode = Node(x[0], x[1], [], x[2])                # Same as UCS but with added heuristic cost to all nodes
        newnode.actionsSoFar.append(x[1])
        fringe.push(newnode, newnode.cost + heur)           # all costs of first level are the original costs

    while not goalFound:
        currnode = fringe.pop()
        currxy = currnode.state

        if currxy not in explored:
            if problem.isGoalState(currxy):                 # check if state is goal
                goalFound = True
                continue

            explored.add(currxy)
            successors = problem.getSuccessors(currxy)

            for x in successors:
                tmpxy = x[0]
                if tmpxy not in explored:                   # all further explored nodes add cost of their parent
                    heur = heuristic(x[0], problem)
                    newnode = Node(x[0], x[1], list(currnode.actionsSoFar), x[2]+currnode.cost)
                    newnode.actionsSoFar.append(x[1])
                    fringe.push(newnode, newnode.cost+heur)

    return currnode.actionsSoFar
    
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
