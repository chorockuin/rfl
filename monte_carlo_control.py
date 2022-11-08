import gym
import matplotlib
import numpy as np
import sys
from collections import defaultdict

from blackjack import BlackjackEnv
import plotting_util

def make_epsilon_greedy_policy(Q, epsilon, nA):
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
        # Todo: np module을 사용하여 적절한 변수를 정의하세요.
        # 모든 action에 동일한 확률을 부여합니다.
        A = np.ones(nA, dtype=float) * epsilon / nA# numpy array
        best_action = np.argmax(Q[observation])# numpy array index
        # max 값을 가지는 index list를 생성합니다.
        tie = np.where(Q[observation] == Q[observation][best_action])[0]# numpy array
        # max 값이 여러개일 경우 tie array에서 랜덤으로 하나를 선택합니다
        best_action = np.random.choice(tie)# numpy array index
        A[best_action] += (1. - epsilon)
        return A
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Epsilon-Greedy policies를 이용한 Monte Carlo Control 함수입니다.
    최적의 Epsilon-Greedy policies를 구합니다.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # state,action에 대한 sum과 state수를 count하는 dictionary입니다.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # state -> (action -> action-value)로 매핑하는 nested dictionary입니다.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Epsilon greedy policy를 생성합니다.
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 에피소드를 생성합니다.
        # episode (list) : list of (state, action, reward) tuples
        episode = []
        state = env.reset()

        # 게임이 종료될 때까지 드로우를 진행합니다.
        while True:
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Todo : For loop를 완성하세요.
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # 처음 등장한 state, action pair의 state를 출력합니다.
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # First-visit MC를 위한 sum-return을 계산합니다.
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Estimated average return을 계산하여 Q에 대입합니다.
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1 
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        
        # The policy is improved implicitly by changing the Q dictionary
    
    return Q, policy

def execute():
    env = BlackjackEnv()
    print('실행 결과 탭을 확인하세요!')

    Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)
    # For plotting: Create value function from action-value function
    # by picking the best action at each state
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    plotting_util.plot_value_function(V, title="Optimal Value Function")
