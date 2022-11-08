import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
from windy_gridworld import WindyGridworldEnv
import plotting_util

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
    
def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control알고리즘입니다.
    Opitmal epsilon greedy policy를 찾습니다.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # state -> (action -> action-value)로 매핑하는 nested dictionary입니다.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 그래프를 그리기 위한 정보를 저장합니다.
    stats = plotting_util.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # epsilon greey policy를 생성합니다.
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # episdoe 진행상황을 출력합니다..
        if (i_episode + 1) % 40 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # 환경을 초기화하고 초기 action을 선택합니다.
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # Episode의 스텝을 종료될 때 까지 반복합니다.
        for t in itertools.count():
            # action으로 다음 state와 reward, done을 계산합니다.
            next_state, reward, done, _ = env.step(action)
            
            # 다음 action을 선택합니다.
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD 업데이트를 진행합니다.
            Q[state][action] += alpha * delta(Q, state, action, reward, next_state, next_action)
            
            # 종료 시 루프를 종료합니다.
            if done:
                break
                
            # 다음 action과 state를 정의합니다.
            action = next_action
            state = next_state
    
    return Q, stats
    
# Todo
def delta(Q, state, action, reward, next_state, next_action, discount_factor=1.0, alpha=.5):
    td_target = reward + discount_factor * Q[next_state][next_action]
    td_delta = td_target - Q[state][action]
    
    return td_delta
    
def execute():
    env = WindyGridworldEnv()
    Q, stats = sarsa(env, 200)
    plotting_util.plot_episode_stats(stats)
    print(f'\n실행 결과 탭을 확인하세요.')
