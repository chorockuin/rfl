import io
import numpy as np
import sys
from gym.envs.toy_text import discrete # gym==0.20.0

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """
    Agent는 MxN 그리드에 존재하고, terminal state에 도달하는 것이 목표입니다.
    terminal state는 좌측 상단 코너와 우측 하단 코너에 존재합니다.
    4x4 그리드는 다음과 같이 표현됩니다.
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    여기서 x는 Agent의 위치, T는 terminal state를 나타냅니다.
    Agent는 상,하,좌,우 4 방향으로 움직일 수 있습니다. (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    grid를 벗어나는 방향으로 움직이게 되면 agent는 제자리에 머물게 됩니다.
    terminal state에 도달하기 전까지 매 step마다 agent는 -1의 reward를 받습니다.
    """

    # 인코딩 방식을 표현합니다.
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape : list = [4,4]):
        # shape element에 대해 tuple 혹은 list를 인자로 받습니다.
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        # nS는 총 state의 크기, nA는 총 action의 크기를 나타냅니다.
        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(nA)}

            is_done = lambda s: s == 0 or s == (nS - 1)
            reward = 0.0 if is_done(s) else -1.0

            # 종료 state의 transistion을 나타냅니다.
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
            ####################### Don't edit the code above #######################
            # Todo: 종료 state의 transition을 참고하여 
            #       종료 state가 아닌 곳에서의 transistion을 정의하세요.
                c = s%MAX_X
                r = int(s/MAX_X)

                P[s][UP] = [(1.0, s-MAX_X, reward, is_done(s-MAX_X))]
                P[s][RIGHT] = [(1.0, s+1, reward, is_done(s+1))]
                P[s][DOWN] = [(1.0, s+MAX_X, reward, is_done(s+MAX_X))]
                P[s][LEFT] = [(1.0, s-1, reward, is_done(s-1))]

                if c == 0:
                    P[s][LEFT] = [(1.0, s, reward, False)]
                
                if c == MAX_X-1:
                    P[s][RIGHT] = [(1.0, s, reward, False)]

                if r == 0:
                    P[s][UP] = [(1.0, s, reward, False)]

                if r == MAX_Y-1:
                    P[s][DOWN] = [(1.0, s, reward, False)]
            ####################### Don't edit the code below #######################
            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()

def execute():
    env = GridworldEnv()
    env._render()
    for k in env.P.keys():
        print(f'env.P[{k}]={env.P[k]}')