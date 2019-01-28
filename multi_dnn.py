# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 17:05:21 2018

@author: CX
"""
import os
import gym
import time
import copy
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from Prioritized_Memory import Memory


class DCNN:
    def __init__(self, network_name, input_state, n_output):
        initializer = tf.contrib.layers.variance_scaling_initializer()
        activation = tf.nn.elu

        with tf.variable_scope(network_name) as scope:
            hidden1 = tf.layers.conv2d(inputs = input_state, filters = 32, 
                                       kernel_size = (4,4), strides = 2, 
                                       padding = 'SAME', activation = activation,
                                       kernel_initializer = initializer)
            hidden2 = tf.layers.conv2d(inputs = hidden1, filters = 64, 
                                       kernel_size = (3,3), strides = 1, 
                                       padding = 'SAME', activation = activation,
                                       kernel_initializer = initializer)
            hidden3 = tf.layers.conv2d(inputs = hidden2, filters = 64, 
                                       kernel_size = (2,2), strides = 1, 
                                       padding = 'SAME', activation = activation,
                                       kernel_initializer = initializer)
            hidden4 = tf.layers.conv2d(inputs = hidden3, filters = 64, 
                                       kernel_size = (2,2), strides = 1, 
                                       padding = 'SAME', activation = activation,
                                       kernel_initializer = initializer)
            flatten = tf.layers.flatten(hidden4)
            hidden5 = tf.layers.dense(flatten, 512, activation = activation, 
                                       kernel_initializer = initializer)
            # Dueling Layer
            S = tf.layers.dense(hidden5, 1,
                                kernel_initializer = initializer)
            A = tf.layers.dense(hidden5, n_output,
                                kernel_initializer = initializer)
            outputs = S + (A - tf.reduce_mean(A, axis = 1, keepdims = True))

        print("A Dueling CNN is created")
        self.Q_values = outputs
        # Collect all the variables in this network
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope = scope.name)
        self.params = {param.name[len(scope.name):]: param for param in params}


class Snake_Agent:
    # This class define the agent and methods for training, testing, etc.
    def __init__(self, env_name, max_iter_or_epis, learning_rate):
        # initialize parameters, tensorflow graph
        self.env_name = env_name
        
        self.env = gym.make(env_name)
        self.env2 = gym.make(env_name)

        self.snake_num = 3
        self.n_actions = 3
        self.eps_max = 0.8
        self.eps_min = 0.1
        self.gamma = 0.99
        self.learning_rate = learning_rate

        # epsilon decay from max to min after this number of iterations
        self.eps_updates = 500000
        # maximum training episodes
        self.max_updates = max_iter_or_epis
        # interval (number of updates) to do a evaluation when training
        self.eval_updates = 10000
        # interval (number of updates) to copy the online network to target network
        self.copy_updates = 400
        self.save_updates = 10000000
        
        self.input_state = tf.placeholder(tf.float32, shape = [None, 20, 20, 1])
        self.input_state_tar = tf.placeholder(tf.float32, shape = [None, 20, 20, 1])
        self.q_network = DCNN("q_network", self.input_state, self.n_actions)
        self.t_network = DCNN("t_network", self.input_state_tar, self.n_actions)

        copy_ops = [t_param.assign(self.q_network.params[name])\
                                   for name, t_param in self.t_network.params.items()]
        self.copy_step = tf.group(*copy_ops)
        
        self.action_taken = tf.placeholder(tf.int32, shape = [None])
        action_onehot = tf.one_hot(self.action_taken, self.n_actions)

        current_value = tf.reduce_sum(self.q_network.Q_values * action_onehot, axis = 1)
        self.target_value = tf.placeholder(tf.float32, shape = [None])
        
        self.abs_error = tf.abs(current_value - self.target_value)
        self.ISWeight = tf.placeholder(tf.float32, [None])
        loss = tf.reduce_mean(self.ISWeight*tf.square(current_value - self.target_value))
        
        self.global_step = tf.Variable(0, trainable = False)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_step = optimizer.minimize(loss, global_step = self.global_step)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def set_file_path(self,path):
        # set the path to save and restore data of network
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.file_path = path + "/saved_data.ckpt"

    def epsilon_greedy_policy(self, q_values, num_updates):
        # epsilon greedy policy with decaying epsilon over training
        eps_step = (self.eps_max-self.eps_min) * num_updates / self.eps_updates
        eps = self.eps_max - eps_step
        eps = max(eps, self.eps_min)

        if np.random.rand() < eps:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(q_values)

    def epsilon_greedy_policy_for_eval(self, q_values):
        # epsilon greedy policy with fixed epsilon for evaluation
        if np.random.rand() < 0.05:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(q_values)

    def greedy_policy(self, q_values):
        # greedy policy for test
        return np.argmax(q_values)

    def train(self):
        # training function with replay memory
        print("Now train with replay memory")

        game_over = True
        n_episodes = 0
        n_iter_this_epi = 0
        n_eat_fruit = [0 for i in range(self.snake_num)]
        n_die = [0 for i in range(self.snake_num)]
        stop_memory = [False for i in range(self.snake_num)]

        # initialize the replay memory
        memory_size = 1000000
        # number of tuples to burn in memory before training
        burn_in_num = 1000
        batch_size = 64

        self.memory = Memory(capacity = memory_size)
        burn_in_done = False # is burn_in finished?

        with tf.Session() as sess:
            if os.path.isfile(self.file_path + '.index'):
                self.saver.restore(sess, self.file_path)
                self.eval_fruit = np.load(self.path + '/eval_fruit.npy')
                self.eval_life_length = np.load(self.path + '/eval_life_length.npy')
                print('Data Restored')
                print(datetime.now())
            else:
                self.init.run()
                self.eval_fruit = np.zeros((1, self.snake_num))
                self.eval_life_length = np.zeros((1, self.snake_num))
                print('Data Initialized')
                print(datetime.now())

            self.copy_step.run()

            while True:
                if game_over or n_iter_this_epi >= 400:
                    if burn_in_done:
                        n_episodes += 1
                    obs = self.env.reset()
                    n_iter_this_epi = 0
                    stop_memory = [False for i in range(self.snake_num)]

                actions = []
                for i in range(self.snake_num):
                    current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                     {self.input_state: [obs[i]]})
                    action = self.epsilon_greedy_policy(current_q_values, self.global_step.eval())
                    actions.append(action)

                n_iter_this_epi += 1
                next_obs, rewards, dones, _ = self.env.step(np.array(actions))

                if burn_in_done:
                    for i in range(self.snake_num):
                        if rewards[i] > 0:
                            n_eat_fruit[i] += 1
                        elif rewards[i] < -2:
                            n_die[i] += 1

                #append this experience to memory
                for i in range(self.snake_num):
                    if not stop_memory[i]:
                        self.memory.store([obs[i], actions[i], rewards[i], next_obs[i], dones[i]])


                stop_memory = copy.deepcopy(dones)
                game_over = all(dones)
                obs = next_obs
                
                # check if memory has been burned in, if finished, start updating
                if not burn_in_done:
                    if self.memory.length < burn_in_num:
                        continue
                    else:
                        # if burn_in is done, restart
                        burn_in_done = True
                        game_over = True
                        continue

                # do memory replay
                    # get the batch from memory
                tree_idx, batch_memory, ISWeights = self.memory.sample(batch_size)
                m_obs, m_action, m_reward, m_next_obs, m_done= batch_memory
                    # get the next_q_values using online network
                next_onl_values = self.q_network.Q_values.eval(
                        feed_dict = {self.input_state: m_next_obs})
                    # get the actions correspond to maximum online values
                next_onl_actions = np.argmax(next_onl_values, axis = 1)
                    # get the next_q_values using target network
                next_tar_values = self.t_network.Q_values.eval(
                        feed_dict = {self.input_state_tar: m_next_obs})

                    # get the next_q_values for target using values from target network
                    # and greedy actions from online network
                next_val4tar = np.zeros(batch_size)
                for k in range(batch_size):
                    next_val4tar[k] = next_tar_values[k, next_onl_actions[k]]
                    # calculate the target value
                target_vals = m_reward+(1 - m_done)*self.gamma*next_val4tar

#                next_q_values = self.t_network.Q_values.eval(
#                        feed_dict = {self.input_state_tar: m_next_obs})
                    # get the maximum next_q_values for updating
#                max_next_q_values = np.max(next_q_values, axis = 1)
                    # get the target value for states and actions
#                target_vals = m_reward+(1 - m_done)*self.gamma*max_next_q_values

                # one training update step
                abs_errors = self.abs_error.eval(feed_dict={
                        self.input_state: m_obs, 
                        self.action_taken: m_action, 
                        self.target_value: target_vals})

                self.train_step.run(feed_dict={
                        self.input_state: m_obs, 
                        self.action_taken: m_action, 
                        self.target_value: target_vals,
                        self.ISWeight: ISWeights})

                self.memory.batch_update(tree_idx, abs_errors)
    
                n_updates = self.global_step.eval()
    
                # save the model every few updates
                if n_updates % self.save_updates == 0:
                    self.saver.save(sess, self.file_path)

                # copy the online network to target network
                if n_updates % self.copy_updates == 0:
                    self.copy_step.run()

                # evaluate the current model every few updates
                if n_updates % self.eval_updates == 0:
                    print(datetime.now())
                    print("current episode:", n_episodes)
                    print("memory length:", self.memory.length)
                    print("updates completed:", n_updates)
                    print("times to eat fruit:", n_eat_fruit)
                    print("times to die:", n_die)
                    self.evaluate()
                    n_eat_fruit = [0 for i in range(self.snake_num)]
                    n_die = [0 for i in range(self.snake_num)]

                if n_updates >= self.max_updates:
                    break

            # save the final model
            self.saver.save(sess, self.file_path)

    def evaluate(self):
        # function to evaluate the model when training
        # record the average reward over 20 episodes using epsilon (0.05) greedy
        n_episodes = 0
        n_iter_this_epi = 0

        all_n_eatfruit = []
        all_n_iters = []

        n_eat_fruit = [0 for i in range(self.snake_num)]
        n_iter = [0 for i in range(self.snake_num)]

        obs = self.env2.reset()

        while True:
            if n_episodes >= 20:
                average_reward = np.mean(np.array(all_n_eatfruit), axis = 0, keepdims = True)
                average_iters = np.mean(np.array(all_n_iters), axis = 0, keepdims = True)

                self.eval_fruit = np.append(self.eval_fruit, average_reward, axis = 0)
                self.eval_life_length = np.append(self.eval_life_length, average_iters, axis = 0)
                
                np.save(self.path + '/eval_fruit.npy',self.eval_fruit)
                np.save(self.path + '/eval_life_length.npy',self.eval_life_length)
                
                print("average times ate fruit over 20 episodes:",average_reward)
                print("average life length over 20 episodes:",average_iters)
                print('')
                break
            
            actions = []
            for i in range(self.snake_num):
                current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                 {self.input_state: [obs[i]]})
                action = self.epsilon_greedy_policy(current_q_values, self.global_step.eval())
                actions.append(action)
            
            n_iter_this_epi += 1
            obs, rewards, dones, _ = self.env2.step(np.array(actions))
            
            for i in range(self.snake_num):
                if not dones[i]:
                    n_iter[i] += 1
            
            for i in range(self.snake_num):
                if rewards[i] > 0:
                    n_eat_fruit[i] += 1
            
            game_over = all(dones)
            
            if game_over or n_iter_this_epi >= 400:
                n_episodes += 1
                n_iter_this_epi = 0
                
                all_n_eatfruit.append(n_eat_fruit)
                all_n_iters.append(n_iter)
                
                obs = self.env2.reset()

                n_eat_fruit = [0 for i in range(self.snake_num)]
                n_iter = [0 for i in range(self.snake_num)]

    def plot_evaluations(self):
        # function to save the plot of evaluation results
        plt.figure()
        plt.plot(self.eval_fruit[1:])
        plt.title("Average Times Eating Fruit_" + self.env_name)
        plt.savefig(self.path + "/Average_Rewards_of_Evaluation.png")
        plt.clf()
        
        plt.figure()
        plt.plot(self.eval_life_length[1:])
        plt.title("Average Life Length_" + self.env_name)
        plt.savefig(self.path + "/Average_Life_Length.png")
        plt.clf()
        
    def play_one_episode(self):
        if not os.path.isfile(self.file_path + '.index'):
            print("No existing weights to play!")
            return
        
        with tf.Session() as sess:
            self.saver.restore(sess, self.file_path)
            
            obs = self.env2.reset()
            self.env2.render()
            i = 0
            while True:
                
                actions = []
                for i in range(self.snake_num):
                    current_q_values = self.q_network.Q_values.eval(feed_dict = 
                                                     {self.input_state: [obs[i]]})
                    action = self.greedy_policy(current_q_values)
                    actions.append(action)
                
                i += 1
                obs, rewards, dones, _ = self.env2.step(np.array(actions))
                self.env2.render()
                time.sleep(0.15)
                
                if i == 70:
                    plt.imshow(obs[1].mean(axis = 2))
                    plt.show()
                
                
                game_over = all(dones)

                if game_over:
                    break
'''
Main Function Part

'''
def main(env_name, file_path, learning_rate, max_iter_or_epis, do_train):

    tf.reset_default_graph()
    # creat the agent
    my_agent = Snake_Agent(env_name, max_iter_or_epis, learning_rate) 
    # set the file path
    my_agent.set_file_path(file_path)

    # do training and evaluation
    if do_train:
        my_agent.train()
        my_agent.plot_evaluations()


if __name__ == '__main__':
    env_name = 'AlphaSnake-v0'
    file_path = "./" + env_name  # path to save the model, videos and plots

    learning_rate = 0.0001
    max_iter_or_epis = 10000000  # maximum training iterations or episodes
    do_train = 1

    main(env_name, file_path, learning_rate, max_iter_or_epis, do_train)
