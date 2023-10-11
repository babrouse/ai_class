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

import util

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
    return  [s, s, w, s, w, w, s, w]

def tinyMazeSearch2(problem):
    """
    I wanted to see if I could make a new function that could solve the tiny maze
    in a differnent path to make sure I understand how it works
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST
    return [w, w, w, w, s, s, e, s, s, w]

def testSearch(problem):
    """
    I'm going to use this to test looking at boards step-by-step
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    return[s]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from game import Directions
    from util import Stack, Queue, PriorityQueue

    n = Directions.NORTH
    e = Directions.EAST
    s = Directions.SOUTH
    w = Directions.WEST

    """
    Graph Search Pseudocode:
    function graph_search(problem, fringe) returns a solution or failure
    closed - an empty set
    fringe - insert (make-node(initial-state(problem)), fringe)

    loop do
        if fringe is empty then return failure
        node -remove-front(fringe)
        if goal-test(problem, state[node]) then return node
        if state[node] not in(closed) then
            add state[node] to closed
            fringe -insert-all(expand(node,problem),fringe)
    end
    """
    # Initialize a set for visited nodes
    closed = set()

    # Initialize a stack to be our fringe and add the start state plus the direction to get there (none)
    # This part had me tricked for a minute since getStartState only returns coordinates, extra parenthesis
    # were needed is all to also include the [].
    fringe = Stack()
    fringe.push((problem.getStartState(), []))

    # Main loop for this algorithm runs as long as the fringe isn't empty (arrived at goal) to which at that
    # point the path that's been filling as we expand is returned. Otherwise, empty path is returned.
    while fringe.isEmpty != True:
        # Initially this sets the state as the start state and the path as the current fringe
        state, path = fringe.pop()

        # Always check first if we're at the goal. If so, we're done and we return the path to the goal
        if problem.isGoalState(state) == True:
            return path
        
        # Here is where we start to avoid revisiting states, if the current state isn't in the closed set, then
        # we add it.
        elif state not in closed:
            closed.add(state)

            # grab possible next states
            successors = problem.getSuccessors(state)
            
            # iterate through them
            for next_state, dir, step_cost in successors:
                
                # here is where we actually make sure not to go back into previously visited nodes
                if next_state not in closed:
                    
                    # If the next node hasn't been visited, we add the direction of movement to path
                    # and push it into the fringe with the next_state
                    add_path = path + [dir]
                    fringe.push((next_state, add_path))
        
        # I put this little guy here so I could watch the fringe expand
        # print(fringe.list)

    # The empty path (failure). I tested this with a maze with no solution and it returns the index error:
    # "popped from empty list.""
    return []

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    "*** YOUR CODE HERE ***"
    # I /think/ this will be implemented similarly but using the queue instead of the stack so hopefully
    # less commenting this time...
    from util import Queue

    closed = set()
    fringe = Queue()
    fringe.push((problem.getStartState(), []))

    print(fringe.list)



    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
