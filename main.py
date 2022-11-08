import bellman_equation_solver
import gridworld
import policy_evaluation
import policy_iteration
import value_iteration
import monte_carlo_prediction
import monte_carlo_control
# import sarsa_on_windy_gridworld
# import q_learning_on_cliff_walking
# import importance_sampling
# import experience_replay

def main():
    bellman_equation_solver.execute()
    gridworld.execute()
    policy_evaluation.execute()
    policy_iteration.execute()
    value_iteration.execute()
    monte_carlo_prediction.execute()
    monte_carlo_control.execute()
    # sarsa_on_windy_gridworld.execute()
    # q_learning_on_cliff_walking.execute()
    # importance_sampling.execute()
    # experience_replay.execute()

if __name__=="__main__":
    main()