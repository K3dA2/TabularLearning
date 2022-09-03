import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")


alpha = 0.1 #learning rate
discount_factor = 0.95
episodes = 35000
episode_count = []
return_count = []
moving_average = []
show = 2000

discrete_obs_size = [20] * len(env.observation_space.high) #make more dynamic
discrete_obs_win_size = (env.observation_space.high - env.observation_space.low)/discrete_obs_size
#print(discrete_obs_win_size)

q_table = np.random.uniform(low=-2, high=0, size = (discrete_obs_size + [env.action_space.n]))
#q_table = np.ones(shape = (discrete_obs_size + [env.action_space.n]))
#print(q_table)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_obs_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range(episodes):
    episode_count.append(episode)
    r = 0
    if episode % show == 0:
        print('Episode: ', episode)
        render = True
    else:
        render = False
    
    discrete_state = get_discrete_state(env.reset())
    done = False

    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        r += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = current_q + alpha * (reward + discount_factor * max_future_q - current_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"Made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state
    return_count.append(r)
    moving_average.append(sum(return_count)/len(return_count))
    #print(return_count)

env.close()

plt.plot(episode_count, moving_average)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.savefig('my_plot_with_random_entries.png')
plt.show()