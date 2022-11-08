import numpy as np
from gridworld import GridworldEnv

def value_iteration(env, theta : float = 0.0001, discount_factor : float = 1.0):
    """
    Value Iteration Algorithm.
    Bellman optimality equation에 근거하여 policy를 업데이트합니다.
    
    Args:
        env: OpenAI env.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    
    def one_step_lookahead(state, V):
        """
        주어진 state에서 가능한 action에 대해 state action value를 계산하여 list로 반환합니다.
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS
        
        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    V = np.zeros(env.nS)
    while True:
        # 현재 state와 best action의 오차입니다.
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Todo: stete action value list를 계산합니다.
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # 오차를 계산합니다.
            delta = max(delta, np.abs(best_action_value - V[s]))
            # value function을 갱신합니다.
            V[s] = best_action_value        
        # Check if we can stop 
        if delta < theta:
            break
    
    # Todo: optimal value function으로부터 deterministic policy를 만듭니다.
    # while 문 안에 구현할 수도 있지만, loop를 분리하는게 더 효율적입니다.
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        #Code here
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    
    return policy, V


def execute():
    env = GridworldEnv()
    
    policy, V = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Reshaped Grid Value Function:")
    print(V.reshape(env.shape))
    print("")