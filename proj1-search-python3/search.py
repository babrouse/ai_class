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
    from util import Stack

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
    closed.add((4, 5)) # test to see if we can 'block' pacman by saying he's already gone left

    # Initialize a stack to be our fringe and add the start state plus the direction to get there (none)
    # This part had me tricked for a minute since getStartState only returns coordinates, extra parenthesis
    # were needed is all to also include the [].
    fringe = Stack()
    start_state = problem.getStartState()
    fringe.push((start_state, []))

    # Main loop for this algorithm runs as long as the fringe isn't empty (arrived at goal) to which at that
    # point the path that's been filling as we expand is returned. Otherwise, empty path is returned.
    while fringe.isEmpty != True:
        # Initially this sets the state as the start state and the path as the current fringe
        state, path = fringe.pop()

        # Always check first if we're at the goal. If so, we're done and we return the path to the goal
        if problem.isGoalState(state) == True:
            return path
        
        # Here is where we start to avoid revisiting states, if the current state isn't in the closed set, then
        # we add it to say we've been here.
        elif state not in closed:
            closed.add(state)

            # grab possible next states. This is an array of triples so we can iterate over this in next step
            successors = problem.getSuccessors(state)
            
            for next_state, dir, step_cost in successors:
                
                # here is where we actually make sure not to go back into previously visited nodes - if next_state
                # was in closed, then we are stuck or we move to the other state
                if next_state not in closed:
                    
                    # If the next node hasn't been visited, we add the direction of movement to path
                    # and push it into the fringe with the next_state
                    add_path = path + [dir]
                    fringe.push((next_state, add_path))
        
        # These little guys were for testing purposes - closed should only grow by one node at a time if working
        # print(fringe.list)
        # print(closed)

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
    start_state = problem.getStartState()
    fringe.push((start_state, []))
    
    # Determine the start states for mediumMaze: (34, 16)

    # I'll test if it's the least cost solution by blocking one path from the get go. Ideal path is cost 68!
    # closed.add((33,16)) # cost of 74 for mediumMaze

    while fringe.isEmpty() != True:
        # The only difference occurs here - since BFS uses a FIFO order, this fring.pop() dequeues the first
        # that was in rather than the LIFO stack of DFS
        state, path = fringe.pop()

        if problem.isGoalState(state) == True:
            return path
        
        elif state not in closed:
            closed.add(state)

            successors = problem.getSuccessors(state)

            for next_state, dir, step_cost in successors:

                if next_state not in closed:
                    add_path = path + [dir]
                    fringe.push((next_state, add_path))
        # print(fringe.list)
        # print(closed)

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    # obviously need the visited stuff still and need to initialize
    closed = set()
    fringe = PriorityQueue()
    start_state = problem.getStartState()

    # closed.add((33,16))

    # PriorityQueue now takes 3 arguments (self, item, priority) so start is obviously priority 0
    # push is fine here since starts as empty PriorityQueue
    fringe.push((start_state, []), 0)

    # have to use heap instead of list - wanted tos ee what this data structure looked like
    print(fringe.heap)

    # basically same while loop except...
    while fringe.isEmpty() != True:
        # The state with highest priority (which is least cost, I believe) and dequeue it
        state, path = fringe.pop()

        if problem.isGoalState(state) == True:
            return path
        
        elif state not in closed:
            closed.add(state)

            successors = problem.getSuccessors(state)

            for next_state, dir, step_cost in successors:

                if next_state not in closed:
                    add_path = path + [dir]

                    # use getCostOfActions to calculate total cost of a path
                    cost_of_path = problem.getCostOfActions(add_path)

                    # add to the fringe with the consideration of how costly the path is
                    # thus the fringe can organize based on cost of a path - I think we need
                    # to use the update method so it can shuffle the priorities to be pushed
                    fringe.update((next_state, add_path), cost_of_path)

        # print(fringe.heap)
    return []

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
    """
    This is going to be the same as the uniform search cost except instead of assigning
    priority 0 at the start, we'll just assign it the heuristic given in the argument.
    As a check, doing this with the nullHeuristic just returns the UCS.
    """
    from util import PriorityQueue

    closed = set()
    fringe = PriorityQueue()
    start_state = problem.getStartState()

    # Start with an initial estimated cost to goal rather than 0 (just below)
    goal_cost = heuristic(start_state, problem)

    # Throw the heuristic in instead of 0
    fringe.update((start_state, []), goal_cost)

    # Loop stays the same
    while fringe.isEmpty() != True:
        state, path = fringe.pop()

        if problem.isGoalState(state) == True:
            return path

        if state not in closed:
            closed.add(state)

            successors = problem.getSuccessors(state)

            for next_state, dir, step_cost in successors:

                add_path = path + [dir]

                # Here we not only calculate the current cost of the path as before but...
                new_path_cost = problem.getCostOfActions(add_path)
                
                # Now we use the heuristic to estimate the cost remaining to get to the goal
                new_goal_cost = heuristic(next_state, problem)
                
                # Priority becomes the actual path cost + the estimated cost to goal
                priority = new_path_cost + new_goal_cost

                fringe.update((next_state, add_path), priority)


        # print(fringe.heap)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
