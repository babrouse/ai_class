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
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            # copy current values so we don't lose them
            prev_vals = self.values.copy()

            # iterate through states
            for state in self.mdp.getStates():
                # initiate very low values for states
                self.values[state] = -1000000000

                # iterate and use the bellman equation
                for action in self.mdp.getPossibleActions(state):
                    value = 0

                    for nstate, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        value += prob * (self.mdp.getReward(state, action, nstate) + self.discount * prev_vals[nstate])
                        
                    if self.values[state] <= value:
                        self.values[state] = value

                if self.values[state] == -1000000000:
                    self.values[state] = 0.0

                        

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
        act_val = 0 # Initialize the original value we'll compute

        # iterate and use the bellman equation here
        for nstate, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # update the value of the action
            act_val += prob * (self.mdp.getReward(state, action, nstate) + self.discount*self.values[nstate])

        return act_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        
        high_bound = -1000000000
        act = None

        for i in self.mdp.getPossibleActions(state):
            q_val = self.computeQValueFromValues(state, i)
            if q_val >= high_bound:
                high_bound = q_val
                act = i
        
        return act
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for i in range(self.iterations):
            state = states[i % len(states)]

            if not self.mdp.isTerminal(state):
                high_bound = -1000000000

                for act in self.mdp.getPossibleActions(state):
                    q_value = self.computeQValueFromValues(state, act)

                    if q_value > high_bound:
                        high_bound = q_value

                self.values[state] = high_bound



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    # ALSO MY CODE!!!!!!!!!!!! :)
    # Need to compute predecessors so making a function for that
    def computePreds(self):
        # dict for predecessors
        preds = {}
        for state in self.mdp.getStates():
            preds[state] = set()

        for state in self.mdp.getStates():
            for act in self.mdp.getPossibleActions(state):
                for nstate, prob in self.mdp.getTransitionStatesAndProbs(state, act):
                    preds[nstate].add(state)
        return preds
        
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Following the algorithm given in the problem statement:
        # Compute predecessors:
        preds = self.computePreds()

        # Start with an empty priority queue
        priQ = util.PriorityQueue()

        # Step 3: Update Priority Queue
        # Iterate over all the states and compute the difference between the
        # current value and the highest possible Q value (hence max)
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state) != True: ####
                high_q = -1000000000

                for act in self.mdp.getPossibleActions(state):
                    q_val = self.computeQValueFromValues(state, act)
                    if q_val > high_q:
                        high_q = q_val
                
                dq = abs(self.values[state] - high_q) # Dairy Queen lol

                priQ.update(state, -dq)

        # Iterate silly!
        for i in range(self.iterations):
            # First check if the priority queue is empty, if it is, break since
            # no need to compute anything
            if priQ.isEmpty() == True:
                break
            
            # pop the highest priority state
            state = priQ.pop()

            # Check to see if the state is terminal since terminal states don't have acts
            # If not terminal, set the value of the state to the max Q-value
            if self.mdp.isTerminal(state) != True:
                high_q = -1000000000

                # Similar to above
                for act in self.mdp.getPossibleActions(state):
                    q_val = self.computeQValueFromValues(state, act)
                    if q_val > high_q:
                        high_q = q_val
                self.values[state] = high_q

            # iterate through preds now
            for pred in preds[state]:
                # Similar to above
                # REMEMBER: we're calculating preds now, not current state
                if self.mdp.isTerminal(pred) != True:
                    max_pred_q = -1000000000

                    for act in self.mdp.getPossibleActions(pred):
                        pred_q = self.computeQValueFromValues(pred, act)
                        if pred_q > max_pred_q:
                            max_pred_q = pred_q
                    dq = abs(self.values[pred] - max_pred_q)

                    # This updates the priority queue if the difference is greater than some
                    # threshold theta, ensuring we don't waste time updating for small changes
                    if dq > self.theta:
                        priQ.update(pred, -dq)
                    







        # # Step 4: Perform Iterations
        # for _ in range(self.iterations):
        #     if priQ.isEmpty():
        #         break
        #     state = priQ.pop()
        #     if not self.mdp.isTerminal(state):
        #         self.values[state] = max([self.computeQValueFromValues(state, a) for a in self.mdp.getPossibleActions(state)])

        #     for p in preds[state]:
        #         if not self.mdp.isTerminal(p):
        #             diff = abs(self.values[p] - max([self.computeQValueFromValues(p, a) for a in self.mdp.getPossibleActions(p)]))
        #             if diff > self.theta:
        #                 priQ.update(p, -diff)