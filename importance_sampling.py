import gym
import matplotlib
import numpy as np
import sys
from collections import defaultdict

from blackjack import BlackjackEnv
import plotting

def make_epsilon_greedy_policy(epsilon, nA, Q=None):
    """
    주어진 epsilon과 Q를 가지고 epsilon-greedy policy를 생성합니다.
    
    Args:
        Q  (dict) : A dictionary that maps from state -> (action -> value (list)).
            Each value is a numpy array of length nA (see below)
        epsilon (float) : The probability to select a random action . float between 0 and 1.
        nA (int) : Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(observation):
        """
        Observation이 주어지면, action policy list를 return하는 함수입니다.
        Args:
            observation : state 
        
        Returns:
            A (list) : A[action_index] = probability
        """
        # 모든 action에 동일한 확률을 부여합니다.
        A = np.ones(nA, dtype=float) * epsilon / nA
        if Q is not None:
            best_action = np.argmax(Q[observation])
            # max 값을 가지는 index list를 생성합니다.
            tie = np.where(Q[observation] == Q[observation][best_action])[0]
            # max 값이 여러개일 경우 그중 하나를 선택합니다
            best_action = np.random.choice(tie)
            A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
    Weighted importance sampling을 적용한 Monte Carlo off-policy 알고리즘입니다.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
        discount_factor: Gamma discount factor.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """
    
    # state -> (action -> action-value)로 매핑하는 nested dictionary입니다.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # weighted importance sampling formula의 denominator입니다.
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 학습하려는 greedy target policy입니다.
    # Todo 1
    target_policy
    
    # Episode를 반복합니다.
    for i_episode in range(1, num_episodes + 1):
        # 진행 상황을 출력합니다.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 에피소드를 생성합니다.
        # episode (list) : list of (state, action, reward) tuples
        episode = []
        state = env.reset()

        # 게임이 종료될 때까지 드로우를 진행합니다.
        for t in range(100):
            # behavior_policy를 이용하여 action probability array를 생성합니다.
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        # Sum of discounted returns
        G = 0.0
        # The importance sampling ratio (the weights of the returns)
        W = 1.0
        # episode를 끝에서부터 진행합니다.
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            # Todo 2
            # time t이후의 discounted return입니다.
            G 
            # Todo 3
            # weighted importance sampling formula의 분모를 업데이트합니다.
            C[state][action]
            # Todo 4
            # action value function을 업데이트합니다.
            Q[state][action] 
            # behavior policy가 생성한 episode의 한 step이 
            # target policy에서 재현할 수 없을 때의 행동을 결정합니다.
            if action !=  np.argmax(target_policy(state)):
            # Todo 5
                
            # Todo 3
            # Importance sampling weight를 업데이트합니다.
            W = 
        
    return Q, target_policy

def execute():
    env = BlackjackEnv()
    print('실행 결과 탭을 확인하세요!')

    # Todo 1
    random_policy = 
    Q, policy = mc_control_importance_sampling(env, num_episodes=500000, behavior_policy=random_policy)
    # For plotting: Create value function from action-value function
    # by picking the best action at each state
    V = defaultdict(float)
    for state, action_values in Q.items():
        action_value = np.max(action_values)
        V[state] = action_value
    plotting.plot_value_function(V, title="Optimal Value Function")