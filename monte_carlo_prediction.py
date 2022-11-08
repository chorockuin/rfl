import gym
import matplotlib
import numpy as np
import pandas as pd
import sys
from collections import defaultdict

from blackjack import BlackjackEnv
import plotting_util

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo policy evaluation 알고리즘입니다.
    Sampling을 이용하여 value function을 추정합니다.
    
    Params:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # state에 대한 sum과 state수를 count하는 dictionary입니다.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    
    # returns_sum과 returns_count로 계산할 value function dictionrary입니다.
    V = defaultdict(float)

    # 에피소드를 n차례 시행합니다.
    for i_episode in range(1, num_episodes + 1):
        # 에피소드 진행상황을 출력합니다.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # 에피소드를 생성합니다.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        
        # 게임이 종료될 때까지 드로우를 진행합니다.
        while True:
            action = policy(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # 게임에서 방문한 모든 state를 저장합니다.,
        # x=sample game, x[0] = (sum_hand, dealer open card, usable_ace) (state)
        states_in_episode = set([x[0] for x in episode])
        # Todo: episode sample을 이용하여 적절한 value function을 출력하세요.
        for state in states_in_episode:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]

    return V

def sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

def execute():
    env = BlackjackEnv()
    print('실행 결과 탭을 확인하세요!')
    
    num_episodes=10000
    print(f'{num_episodes = }')
    V_10k = mc_prediction(sample_policy, env, num_episodes)
    plotting_util.plot_value_function(V_10k, title=f'{num_episodes} steps')