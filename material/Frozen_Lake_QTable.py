
import gym
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import os


env = gym.make('FrozenLake-v0')

def find_max_action(reward_vector):
    max_reward_data = [i for i, e in enumerate(reward_vector) if e == max(reward_vector)]
    if len(max_reward_data) == 1:
        selected_choice = max_reward_data[0]
    else:
        selected_choice = random.choice(max_reward_data)
    return selected_choice

def select_action_to_take(training_mode, episode, num_choices, reward_vector):
    if training_mode:
        epsilon = np.exp(-.01*episode)

        ##############################################################################
        #In training mode, you check if you random # is > epsilon
        #If the random number is > epsilon you choose the best option
        #If the random number is < epsilon, you take a guess
        #By the time you get to episode 100, epsilon is approx 0.367 so less like to guess
        #More likely to guess in the beginning
        ##############################################################################
        random_number = random.random()

        if random_number < epsilon:
            selected_action = np.random.choice(num_choices, 1)[0]
        else:
            selected_action = find_max_action(reward_vector)

    else: #not training_mode
        selected_action = find_max_action(reward_vector)
    return selected_action


def playGame(training_mode, Q, global_Q_list, alpha=0.8, gamma=0.95, num_episodes=2000, max_steps_per_episode=99):
    #Note: To play in random mode, select training_to be False. It will set the Q matrix to be 0 in state x action space.
    #To train, pass in training_mode = True it will then update the Q matrix
    #To play a Q, pass in a predetermined_Q and training mode to be False.


    #create lists to contain total rewards and steps per episode
    episode_rewards = []
    episode_steps = []
    for episode in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()

        episode_reward = 0
        step = 0

        while step < max_steps_per_episode:
            ##########################################################################################
            #Choose an action by greedily (with noise) picking from Q table
            # note env.action_space.n = 4
            # With every new iteration the random component to add is smaller
            ##########################################################################################
            a = select_action_to_take(training_mode=training_mode, episode=episode, num_choices=env.action_space.n, reward_vector=Q[s, :])

            ####################################################################################
            #Get new state and reward from environment
            ####################################################################################
            #env.step(a) returns 4 values #1) observation, 2) reward 3)done(boolean) 4) info. Note that in action space: Left = 0, down = 1,  right =2 , up =3. Also, visualize with env.render()
            observation, reward, done, info = env.step(a)

            #Update Q-Table with new knowledge
            if(training_mode): #if we're in training mode then update. Else don't touch the Q.
                global_Q_list.append(Q.copy())
                Q[s,a] = Q[s,a] + alpha*(reward + gamma*np.max(Q[observation, :]) - Q[s, a])

            episode_reward += reward
            s = observation

            if done:
                episode_rewards.append(episode_reward)
                episode_steps.append(step)
                break

            step += 1
        if (done!= True):
            episode_rewards.append(episode_reward)
            episode_steps.append(step)


    results_df = pd.DataFrame(list(zip(episode_steps, episode_rewards)), columns = ['num_steps', 'game_reward'])
    results_df['loss_time_out'] = [((results_df.iloc[x]['num_steps'] == max_steps_per_episode) and (results_df.iloc[x]['game_reward'] == 0)) for x in range(results_df.shape[0])]
    results_df['loss_hole'] = [((results_df.iloc[x]['num_steps'] < max_steps_per_episode) and (results_df.iloc[x]['game_reward'] == 0)) for x in range(results_df.shape[0])]
    return Q, results_df, global_Q_list


def make_plot(game_result_df, plot_title):
    # plot the rewards and number of steps over all training episodes
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(game_result_df['game_reward'], '-g', label='reward')
    ax1.set_yticks([0, 1])
    ax2 = ax1.twinx()
    ax2.plot(game_result_df['num_steps'], '+r', label='num_steps')
    ax1.set_xlabel("episode")
    ax1.set_ylabel("reward")
    ax2.set_ylabel("step")
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.title(plot_title)
    plt.show()
#####################################################################################
#Training Section
#####################################################################################
start_time = time.time()
training_Q, training_game_result_df, global_Q_list = playGame(training_mode=True, Q=np.zeros([env.observation_space.n,env.action_space.n]), global_Q_list=[])
end_time = time.time()

#np.save(os.path.expanduser("~/gitRepos/qlearning/test_code/part0/video_data/Q_evolution"), np.array(global_Q_list))

print("Out of the " + str(training_game_result_df.shape[0]) + " games, we have won: " + str(round(100*sum(training_game_result_df['game_reward'])/training_game_result_df.shape[0], 2)) + "%")
print("Training took: " + str(int(end_time-start_time)) + " seconds")
make_plot(training_game_result_df, "Training Results")


print("Final Q-Table Values")
print(np.round(training_Q, 3))


#####################################################################################
#Playing Random Mode
#####################################################################################
start_time = time.time()
playing_Q, playing_game_result_df, global_Q_list = playGame(training_mode=False, Q = np.zeros([env.observation_space.n,env.action_space.n]), global_Q_list=[])
end_time = time.time()

print("Out of the " + str(playing_game_result_df.shape[0]) + " games, we have won: " + str(round(100*sum(playing_game_result_df['game_reward'])/playing_game_result_df.shape[0], 2)) + "%")
print("Training took: " + str(int(end_time-start_time)) + " seconds")
make_plot(playing_game_result_df, "Random Results")

print("Final Q-Table Values")
print(np.round(playing_Q, 3))


#####################################################################################
#Playing Trained Model Mode
#####################################################################################
training_mode = False
start_time = time.time()
playing_Q, playing_game_result_df, global_Q_list = playGame(training_mode=False, Q=training_Q , global_Q_list=[])
end_time = time.time()

print("Out of the " + str(playing_game_result_df.shape[0]) + " games, we have won: " + str(round(100*sum(playing_game_result_df['game_reward'])/playing_game_result_df.shape[0], 2)) + "%")
print("Training took: " + str(int(end_time-start_time)) + " seconds")
make_plot(playing_game_result_df, "Test Results")

print("Final Q-Table Values")
print(np.round(playing_Q, 3))
