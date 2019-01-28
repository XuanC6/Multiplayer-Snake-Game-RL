import gym
import time
#import numpy as np
#import gym_snakegame
#from gym import wrappers
import matplotlib.pyplot as plt


env = gym.make('AlphaSnake-v0')
#env = wrappers.Monitor(env, 'my_awesome_dir',force=True)

for i in range(100000):
    obs = env.reset()
    env.render()
    time.sleep(0.2)
#    plt.imshow(obs[0].mean(axis = 2))
#    plt.show()
    for t in range(1000):
        actions = env.action_space.sample()
        a = input('action0: ')
        actions[0] = a
        
        obs, reward, dones, info = env.step(actions)
#        print(reward)
##        plt.imshow(obs[1].mean(axis = 2))
##        plt.show()
##        plt.imshow(obs[2].mean(axis = 2))
##        plt.show()
        env.render()
        time.sleep(0.2)
#        plt.imshow(obs[0].mean(axis = 2))
        plt.show()

        if all(dones):
            print('episode {} finished after {} timesteps'.format(i, t))
            break