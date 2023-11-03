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


from util import manhattanDistance, euclideanDistance
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print(successorGameState)
        # print(newPos)
        # print(newFood)
        # print(newGhostStates)
        # print(newScaredTimes)

        # Define an evaluation score
        pot_score = successorGameState.getScore()

        # Make a list of the current food state:
        food_list = newFood.asList()

        ## Define how to prioritize food
        # If there's still food..
        if len(food_list) > 0:
            food_dists = []

            # Calculate the distance to food
            for i in range(len(food_list)):
                dist = manhattanDistance(newPos, food_list[i])

                food_dists.append(dist)
            
            # Find the closest food
            if len(food_dists) > 0:
                close_food = min(food_dists)

            # Tell the agent that closest food is preferable
            pot_score += 1 / close_food

        for i, ghost_state in enumerate(newGhostStates):
            g_dist = manhattanDistance(newPos, ghost_state.getPosition())
            # print(g_dist)
            if g_dist <= 5 and newScaredTimes[i] == 0:
                pot_score += -69

        # print(newGhostStates[0].getPosition())
                

        return pot_score
        # return successorGameState.getScore()

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, 0)[1]

    # This will need to be recursive so I'm going to create a bunch of functions (a whopping 3)    
    def maxi(self, state, depth, agentIndex):
        # Initialize the lower bound, next action, and agent
        low_bound = -10000000000        
        nact = None
        agent = agentIndex

        # Iterate through possible actions
        for i in state.getLegalActions(agent):
            nstate = state.generateSuccessor(agent, i) # next state

            pot_score = self.minimax(nstate, depth, 1)[0] # run minimax and pull the returned score
            if pot_score > low_bound:
                low_bound = pot_score
                nact = i
        return low_bound, nact

    def mini(self, state, depth, agentIndex):
        high_bound = 100000000000
        nact = None

        agent = agentIndex

        for i in state.getLegalActions(agent):
            nstate = state.generateSuccessor(agent, i)

            if agentIndex == state.getNumAgents() - 1:
                pot_score = self.minimax(nstate, depth + 1, 0)[0]

            else:
                pot_score = self.minimax(nstate, depth, agentIndex + 1)[0]

            if pot_score < high_bound:
                high_bound = pot_score
                nact = i
        return high_bound, nact


    def minimax(self, state, depth, agentIndex):
        # check to see if the agent is at the depth of the tree
        maxDepth = self.depth
    
        # Check to see if the game has won or lost
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        # Check to see if the agent is at the max depth of the minimax tree
        # This could be combined with above but they're technically different in my mind
        elif depth == maxDepth:
            return self.evaluationFunction(state), None

        elif agentIndex == 0:
            # Execute the maxmimizing function as pacman is supposed to maximize
            return self.maxi(state, depth, agentIndex)
        
        else:
            return self.mini(state, depth, agentIndex)
        
        """ 
        REFACTORED THIS INTO maxi AND mini FUNCTIONS
        Now lets consider the agent, Pacman has index 0 so we need to see if we're maxing,
        so check if it's pacman and if it is, we'll work on maxing.
        elif agentIndex == 0:
            # Start with the max score being very small and no best action yet
            low_bound = -10000000000
            nact = None         # Next action
            for i in state.getLegalActions(agentIndex):

                nstate = state.generateSuccessor(agentIndex, i) # Next state

                 # move to next depth and perform the minimax function with ghosts as the agent
                value = self.minimax(nstate, depth, 1)[0]

                if value > low_bound:
                    low_bound = value # Close in on a new minimum score
                    nact = i
            return low_bound, nact
        
        
        # Same stuff but for a minimizing agent - the spooky ghosts
        elif agentIndex >= 1:
            high_bound = 10000000000 # Initial very high score
            nact = None

            # basically same as above
            for i in state.getLegalActions(agentIndex):
                nstate = state.generateSuccessor(agentIndex, i)

                # Check to see if this is the last ghost:
                if agentIndex == state.getNumAgents() - 1:
                    # If last ghost, increase depth and move onto pacman (agent 0)
                    value = self.minimax(nstate, depth + 1, 0)[0]

                    # Otherwise move onto the next ghost
                else:
                    value = self.minimax(nstate, depth, agentIndex + 1)[0]
                
                if value < high_bound:
                    high_bound = value
                    nact = i
            return high_bound, nact
        """


                    


        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    A, B = -10000000000, 100000000000

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        return self.minimax(gameState, 0, 0, A, B)[1]

    # Copy/pasted a lot of stuff from minimax
    
    def maxi(self, state, depth, A, B):
        low_bound = -10000000000
        nact = None
        agent = 0
        for i in state.getLegalActions(agent): # iterate through actions
            nstate = state.generateSuccessor(agent, i) # next state

            pot_score = self.minimax(nstate, depth, 1, A, B)[0] # run minimax and pull the returned score
            if pot_score > B:
                low_bound = pot_score
                A = max(A, pot_score)
                nact = i
        return low_bound, nact

    def mini(self, state, depth, agentIndex, A, B):
        high_bound = 100000000000
        nact = None

        for i in state.getLegalActions(agentIndex):
            nstate = state.generateSuccessor(agentIndex, i)

            if agentIndex == state.getNumAgents() - 1:
                pot_score = self.minimax(nstate, depth+1, 0, A, B)[0]

            else:
                pot_score = self.minimax(nstate, depth, agentIndex + 1, A, B)[0]

            if pot_score < high_bound:
                high_bound = pot_score
                B = min(B, pot_score)
                nact = i
        return high_bound, nact
    
    # def max_value(state, A, B, agentIndex):
    #     v = -100000000000
    #     agent = agentIndex
    #     for i in state.getLegalActions(agent):
    #         nstate = state.generateSuccessor(agent, i)
    #         v = max(v, )


    # This will need to be recursive so I'm going to create a bunch of functions
    def minimax(self, state, depth, agentIndex, A, B):
        # check to see if the agent is at the depth of the tree
        maxDepth = self.depth
    
        # Check to see if the game has won or lost
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        # Check to see if the agent is at the max depth of the minimax tree
        # This could be combined with above but they're technically different in my mind
        elif depth == maxDepth:
            return self.evaluationFunction(state), None

        elif agentIndex == 0:
            # Execute the maxmimizing function as pacman is supposed to maximize
            print(A, B)
            return self.maxi(state, depth, A, B)
        
        else:
            print(A, B)
            return self.mini(state, depth, agentIndex, A, B)
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction