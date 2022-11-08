import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
from collections import defaultdict
from cliff_walking import CliffWalkingEnv
import plotting_util
from memory import ReplayBuffer

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    앞 단원에서 실습한 epsilon greedy policy입니다.
    https://academy.elice.io/courses/15876/lectures/131296/materials/9
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control 알고리즘입니다.
    Behavior policy로 epsilon-greedy policy를 사용하여 optimal policy를 찾습니다.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and       episode_rewards.
    """
    
    # state -> (action -> action-value)로 매핑하는 nested dictionary입니다.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    buffer = ReplayBuffer(2500, 1)

    # 그래프를 그리기 위한 정보를 저장합니다.
    stats = plotting_util.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # epsilon greey policy를 생성합니다.
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # episdoe 진행상황을 출력합니다..
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
        
        # environment를 초기화합니다.
        state = env.reset()
        
        # Episode의 스텝을 종료될 때 까지, 또는 step 50까지 반복합니다.
        for t in range(100):
            
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            
            # Todo 1
            buffer.push(state, action, next_state, reward, done)

            if done:
                break
            
            state = next_state
    
        # Todo 2
        for i in range(20):
            for sample in buffer.sample_batch(100):
                Q[sample.state][sample.action] += alpha * delta(Q, sample.state, sample.action, sample.next_state, sample.reward)

    return Q, stats

# Todo 3
def delta(Q, state, action, next_state, reward, discount_factor=1.0):
    best_next_action = np.argmax(Q[next_state])
    td_target = reward + discount_factor * Q[next_state][best_next_action]
    td_delta = td_target - Q[state][action]
    
    return td_delta

def execute():
    env = CliffWalkingEnv()
    Q, stats = q_learning(env, 1000)
    print('')
    for key in Q.keys():
        print(f'Q[{key}]={Q[key]}')