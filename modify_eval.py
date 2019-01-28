# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:21:41 2018

@author: CX
"""
import numpy as np
import matplotlib.pyplot as plt

env_name = 'AlphaSnake-v0'
path = "./" + env_name

eval_fruit = np.load(path + '/eval_fruit.npy')
eval_life_length = np.load(path + '/eval_life_length.npy')

plt.figure()
plt.plot(eval_fruit[1:])
plt.title("Average Times Eating Fruit_AlphaSnake-v0")
plt.show()

plt.figure()
plt.plot(eval_life_length[1:])
plt.title("Average Life Length_AlphaSnake-v0")
plt.show()
#env_name = 'AlphaSnake-v0'
#path = "./" + env_name
#
#eval_fruit = np.load(path + '/eval_fruit.npy')
#eval_life_length = np.load(path + '/eval_life_length.npy')
#
#eval_fruit = eval_fruit[0:201]
#eval_life_length = eval_life_length[0:201]
#
#np.save(path + '/eval_fruit.npy',eval_fruit)
#np.save(path + '/eval_life_length.npy',eval_life_length)