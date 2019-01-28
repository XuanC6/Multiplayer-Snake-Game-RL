#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:59:05 2018

@author: xuan
"""
#import tensorflow as tf
from multi_dnn import Snake_Agent

'''
Main Function Part

'''
def main(env_name, file_path, learning_rate, max_iter_or_epis, do_train):

    #tf.reset_default_graph()
    # creat the agent
    my_agent = Snake_Agent(env_name, max_iter_or_epis, learning_rate) 
    # set the file path
    my_agent.set_file_path(file_path)
    my_agent.play_one_episode()


if __name__ == '__main__':
    env_name = 'AlphaSnake-v0'
    file_path = "./" + env_name  # path to save the model, videos and plots

    learning_rate = 0.0001
    max_iter_or_epis = 100000  # maximum training iterations or episodes
    do_train = 0

    main(env_name, file_path, learning_rate, max_iter_or_epis, do_train)
