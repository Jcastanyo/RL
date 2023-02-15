'''

    CODE FOR TESTING RL ALGORITHMS WHILE STUDYING SUTTON AND BARTO BOOK AND DEEP REINFORCEMENT LEARNING HANDS-ON BOOK

    Works with gym 0.26.2 version

'''


import numpy as np
import gym
import time


'''
    Important in case you did not know:

        env.P returns the transition matrix with the probabilities, next_actions, rewards and done_info from each state.

'''

# class to wrapp the frozen_lake env from gym and implement policy_evaluation, policy_iteration and value_iteration.
class FrozenLake:
    def __init__(self, size=8):
        """
            Definition of frozen_lake v0 env from gym (size and slippery mode), V function and policy pi.

            ARGS:
                -INPUT:
                    - size: defines the map_name of the env (size of the env: 4x4, 8x8)
        """

        self.size = size
        # max number of iterations in policy evaluation.
        self.max_iter = 100

        # definition and reset of the env.
        self.env = gym.make('FrozenLake-v1', map_name=f"{size}x{size}", is_slippery=False, render_mode='human')
        # mandatory to reset the env at the beggining.
        self.env.reset()
        
        # definition of V function and policy pi.
        self.env_nS = self.env.observation_space.n  # num of states
        self.env_nA = self.env.action_space.n  # num of actions
        self.state_value = np.zeros(self.env_nS)
        self.policy = np.zeros(self.env_nS, dtype=int)
        
    
    def render(self, env, max_steps=100):

        '''
            Function used to visualize the agent solving the environment.

            ARGS:
                -INPUT:
                    - max_steps: total number of steps to solve the env.
        '''

        episode_reward = 0
        ob = env.reset()

        for t in range(max_steps):

            env.render()
            time.sleep(0.25)
            # get the action from the policy
            # to solve a bug because sometimes ob is a tuple containing the state
            # and the prob. of transition and sometimes is only the state
            if isinstance(ob, tuple):
                ob = ob[0]
            a = self.policy[ob]
            # apply the action and get next_state, reward, done_info (true/false)
            ob, rew, done, _, _ = env.step(a)
            # accumulate reward
            episode_reward += rew

            # ends when the agent achieves the terminal state.
            if done:
                break

        env.render()
        
        if not done:
            print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
        else:
            print("Episode reward: %f" % episode_reward)


    def bellman_equation(self, state, gamma):

        '''
            Function used to apply the bellman equation for the policy
            evaluation algorithm.

            ARGS:
                -INPUT:
                    - state: agent's state in the env.
                    - gamma: discount factor of the bellman equation.
        '''

        
        v_new = 0

        # apply the policy
        a = self.policy[state]

        # IMPORTANT: the envs can be deterministic or stochastic, then, the 
        # implementation is prepared for both types of env. In stochastic envs,
        # one action may have different transitions depending on the dynamic of the
        # environment.
        transitions = self.env.P[state][a]
        
        # stochastic env:
        #                   -----> s1 with probability_1
        # s -> policy -> a  -----> s2 with probability_2
        #                   -----> s3 with probability_3 ...
        for t in transitions:
            p, s_1, r, done = t
            v_new += p * (r + gamma * self.state_value[s_1])
 
        return v_new


    def policy_evaluation(self, threshold, gamma):

        '''
            (in-place version) Implementation of policy evaluation. In th in-place
            version, we only use one table for V, the old values are overwritter during
            the sweep for all the states. (faster implementation).

            ARGS:
                -INPUT:
                    - treshold: min. error to achieve during V improvement.
        '''

        # run for max_iter iterations
        for iter in range(self.max_iter):

            v_max = 0
            
            # run over each state
            for state in range(self.env_nS):

                # save the old value V[s]
                v_old = self.state_value[state]
                
                # update V[s] with the bellman equation
                self.state_value[state] = self.bellman_equation(state, gamma)

                # calculate the error between the old V[s] and the new V[s]
                v_error = abs(self.state_value[state] - v_old)
                # get the max error of V.
                v_max = max(v_max, v_error)
    
            # when the max error of V is smaller than the threshold, we consider
            # that the V function is optimal.
            if v_max <= threshold:
                break
            


    def policy_improvement(self, gamma):

        '''
            Implementation of policy improvement algorithm.
        '''
        # bool variable to end the algorithm
        bool = True

        for state in range(self.env_nS):

                # apply policy to save old_action pi(s)              
                old_action = self.policy[state] 

                # run over actions
                action_values = []
                for a in range(self.env_nA):

                    # taking into account stochastic envs.
                    t_values = 0
                    transitions = self.env.P[state][a]
                    for t in transitions:
                        p, s_1, r, done = t
                        t_values += p * (r + gamma * self.state_value[s_1])
                    
                    action_values.append(t_values)
                        
                # in policy improvement we calculate the values for each action in each state,
                # and the we update the policy with the action with higher value.
                self.policy[state] = np.argmax(action_values)
                
                # ends when the policy stops improving. 
                # (same policy in two consecutive iterations)
                if old_action != self.policy[state]:
                    bool = False

        
        return bool

    
    def policy_iteration(self):

        '''
            Implementation of policy iteration. The algorithm first obtains V from an
            arbitrary policy pi with policy_evaluation, and then improves the policy
            pi using the previously calculated V with policy_improvement function.
             (repeat until convergence)
        '''

        self.policy_stable = False
        iter = 0

        while self.policy_stable == False:
            
            self.policy_evaluation(threshold=1e-3, gamma=0.9)
            
            self.policy_stable = self.policy_improvement(gamma=0.9)
            iter += 1
        
        print(f"Number of iterations: {iter}")


    def value_iteration(self, threshold, gamma):

        '''
            Value iteration algorithm combines policy_evaluation 
            and policy_improvement in one sweep. To do it, in each
            state takes the maximum value corresponding to the optimal
            action. Value iteration does not calculate the policy, but
            it can be easily obtained by saving the actions corresponding
            to the highest values for each state.
        '''

        iter = 0

        for _ in range(self.max_iter):

            v_min = 0
            
            for state in range(self.env_nS):

                action_values = []
                v_old = self.state_value[state]
                
                for a in range(self.env_nA):
                    t_values = 0

                    transitions = self.env.P[state][a]
                    
                    for t in transitions:
                        p, s_1, r, _ = t
                        t_values += p * (r + gamma * self.state_value[s_1])

                    action_values.append(t_values)
                
                self.state_value[state] = max(action_values)
                self.policy[state] = np.argmax(action_values)

                v_error = abs(self.state_value[state] - v_old)
                v_min = max(v_min, v_error)

            iter += 1

            if v_min <= threshold:
                break
        
        print(f"Number of iterations: {iter}")


def main():
    
    frozen_lake_env = FrozenLake()

    # good practice to test the agent in a different environment (testing env)
    frozen_lake_env_test = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human')

    print(f"-----------------POLICY ITERATION RUNNING-----------------")
    frozen_lake_env.policy_iteration()
    frozen_lake_env.render(frozen_lake_env_test, 100)
    print(f"-----------------VALUE ITERATION RUNNING-----------------")
    frozen_lake_env.value_iteration(1e-3, 0.9)
    frozen_lake_env.render(frozen_lake_env_test, 100)


if __name__ == '__main__':
    main()