import numpy as np

def execute():
    print(bellman_equation_solver())

def bellman_equation_solver():
    # state의 전이 확률을 나타내는 matrix입니다.
    # matrix의 m번째 행의 n번째 열의 요소는 state m에서 n으로 전이될 확률을 나타냅니다.
    # Todo: 프로그래밍 목표의 그림을 참고하여 transition_matrix에 알맞은 값으로 대체합니다.
    transition_matrix = np.array([[ 0.7,  0.3,  0 , 0,  0,  0,  0],
                                  [ 0.5,  0,  0.5 , 0,  0,  0,  0],
                                  [ 0.1,  0.8,  0 , 0.1,  0,  0,  0],
                                  [ 0,  0,  0.1 , 0,  0.1,  0.8,  0],
                                  [ 0,  0,  0 , 0,  0,  0,  1.0],
                                  [ 0,  0.5,  0.4 , 0,  0,  0,  0.1],
                                  [ 0,  0,  0 , 0,  0,  0,  0]])
    
    # Todo: numpy module을 사용하여 transition matrix와 크기가 동일한 identity matrix를 정의합니다.
    I = np.identity(len(transition_matrix))

    # 리워드 벡터를 np.array 함수를 사용하여 정의합니다.
    reward_vector = np.array([-2, 1, 2, 1, 10, -1, 0])
    
    # Todo: identity matrix에 transition matrix를 빼고, 역행렬을 계산합니다.
    # square matrix A가 numpy array로 주어졌을 때, A의 역행렬은 np.linalg.inv(A)로 쉽게 구할 수 있습니다.
    subtracted_matrix = I - transition_matrix
    inverse_matrix = np.linalg.inv(subtracted_matrix)
    print(inverse_matrix)
    
    # Todo: 계산한 역행렬에 reward vector를 곱해서 state value function값을 벡터 형태로 나타냅니다.
    state_value_function_vector = inverse_matrix@reward_vector
    
    # state value function vector를 int 타입으로 변환하여 소수점 이하를 버립니다.
    return state_value_function_vector.astype(int)

if __name__ == "__main__":
    main()