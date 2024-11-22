import numpy as np

def pia(gamma, R, P):
    # States
    n = P.shape[0]
    # Actions
    m = P.shape[2]

    # Initialize the pi matrix n*m
    pi = np.zeros((n, m))
    for state in range(n):
        # Look up actions that can be taken
        actions = []
        for action in range(m):
            for state_p in range(n):
                # If P(s,s',a) > 0, this action is a possibility
                if P[state][state_p][action] > 0:
                    actions.append(action)
                    break

        # Assign the probabilities equally for the actions
        for action in actions:
            pi[state][action] = 1 / len(actions)
        
    return pi, V