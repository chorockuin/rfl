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

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. 
    반복적으로 policy의 value를 계산하고 업데이트합니다.
    업데이트된 policy가 optimal policy에 도달하면 종료됩니다.
    
    Args:
        env: The OpenAI environment.
        policy_eval_fn (ptr) : Policy evaluation function를 가르키는 포인터입니다.
                               (policy, env, discount_factor)를 input으로 받습니다.
        discount_factor (float) : gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        반환하는 policy는 optiaml policy입니다.
        policy는 [env.nS, env.nA] 형태의 matrix이고, policy[s][a]는 π(a|s)를 나타냅니다.
        V는 optimal policy의 state value function의 list representation입니다.
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
                # Todo : state s에서 action a를 실행할 때의 value를 반복문 형태로 더할 때, 
                # 반복문의 한 iteration에 들어갈 알맞은 공식을 Bellman equation을 참고하여 적으세요..
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        # 현재 policy를 계산합니다.
        V = policy_eval_fn(policy, env, discount_factor)
        
        # Policy가 변하면 False, 아니면 true를 반환하는 boolean입니다.
        policy_stable = True
        
        for s in range(env.nS):
            # Todo: state s에서 가장 높은 확률을 가지는 action을 선택합니다.
            chosen_a = np.argmax(policy[s])
            
            # Todo
            # Find the best action by one-step lookahead
            # 두 action의 값이 같다면 임의로 하나를 반환합니다.
            action_values = one_step_lookahead(s, V)
            # Todo: numpy 모듈에서 적절한 메소드를 이용하여 list변수 action_values에서 best action을 반환하도록 하세요.
            best_a = np.argmax(action_values)
            
            # Todo: policy를 업데이트합니다.
            policy[s] = np.eye(env.nA)[best_a]

            # Todo
            if chosen_a != best_a:
                policy_stable = False
        
        # 모든 state에 대하여 chosen_a와 best_a가 같다면 optimal policy를 반환합니다.
        if policy_stable:
            return policy, V

def execute():
    env = GridworldEnv()
    policy, V = policy_improvement(env)
    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Reshaped Grid Value Function:")
    print(V.reshape(env.shape))
    print("")