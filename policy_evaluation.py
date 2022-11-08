import numpy as np
from gridworld import GridworldEnv

def policy_eval(policy, env, discount_factor : float = 1.0, theta : float = 0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy (list) : [S, A] shaped matrix representing the policy. policy[s][a] equals to π(a|s) in textbook
        env (gym env) : OpenAI env. env.P represents the transition probabilities of the environment.
                        env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                        env.nS is a number of states in the environment. 
                        env.nA is a number of actions in the environment.
        discount_factor (float) : Gamma discount factor.
        theta (float) : We stop evaluation once our value function change is less than theta for all states.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    
    # 초기 state value를 0으로 설정합니다.
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # 각각의 state마다 full-width backup을 시행합니다.
        for s in range(env.nS):
            v = 0
            # 가능한 action에 대해 iteration합니다.
            for a, action_prob in enumerate(policy[s]):
                # 각각의 action에 대해 가능한 state를 구하여 value function에 도입합니다.
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Todo : V[s]를 구하기 위한 적절한 v를 구하세요.
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break

    return np.array(V)

def execute():
    env = GridworldEnv()
    uniform_policy = np.ones([env.nS, env.nA])/env.nA
    v = policy_eval(uniform_policy, env)
    v = v.reshape(env.shape)
    print(v)