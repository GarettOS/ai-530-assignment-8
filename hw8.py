def pia(gamma, R, P):
    # States
    n = P.shape[0]
    # Actions
    m = P.shape[2]

    # Initialize the pi matrix n*m
    pi = [[0 for i in range(m)] for j in range(n)]
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
        
    # Initialize the value function
    V = [0 for x in range(n)]

    # Repeat until convergence
    termination_criterion = 0.001
    converged = False
    while not converged:
        converged = True # assume is true until a policy changes

        # Policy Evaluation
        # Loop until change is negligable
        while True:
            # Keep track of how much V changes
            value_change = 0
            # Compute V[s] for all states 
            for state in range(n):
                # Track the old value 
                old_val = V[state]
                total = 0
                for action in range(m):
                    # If action is possible
                    if pi[state][action] > 0:
                        action_val = 0
                        for state_p in range(n):
                            # Bellman equation
                            action_val += P[state][state_p][action] * (R[state][state_p][action] + (gamma * V[state_p]))
                        # Add to expected 
                        total += pi[state][action] * action_val
                # update value at this state
                V[state] = total
                # update the amount of change
                if (old_val - V[state]) > value_change:
                    value_change = old_val - V[state]
                elif (V[state] - old_val) > value_change:
                    value_change = V[state] - old_val
            if value_change < termination_criterion:
                break
        
        # Policy improvment
        for state in range(n):
            old_actions = pi[state][:]
            # A[s]
            actions = []
            # Find the max action for this state
            for action in range(m):
                total = 0
                for state_p in range(n):
                    # Bellman equation
                    total += P[state][state_p][action] * (R[state][state_p][action] + (gamma * V[state_p]))
                actions.append(total)
            
            # Find the max value 
            max_value = max(actions)
            
            # Choose actions with max value
            max_actions = []
            for i in range(len(actions)):
                if actions[i] == max_value:
                    max_actions.append(i)
            
            # Update policy to use max actions

            # Reset current policy for this state
            pi[state] = [0 for i in range(m)]
            # Replace with new actions
            for a in max_actions:
                pi[state][a] = 1 / len(max_actions)

            # If there was a change, it hasn't converged
            if pi[state] != old_actions:
                converged = False

    return pi, V