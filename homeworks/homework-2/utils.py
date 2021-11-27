import random

import torch
import numpy as np
import matplotlib.pyplot as plt


def state_to_layers(state):
    return np.array([
        (state == 1).astype(float), 
        (state == -1).astype(float), 
        (state == 0).astype(float)
    ])
        

def get_action_from_model(model, state):
    model.eval()
    state = state_to_layers(state)
    state = torch.FloatTensor([state])
    action = model(state).detach().max(1)[1][0].item()
    return action
    

def get_action(model, state, game_hash, epsilon):
    if random.random() < (1 - epsilon):
        action = get_action_from_model(model, state)
    else:
        action = random.choice(is_available(game_hash, True))
    return action


def is_available(s, available=True):
    string_array = np.array(list(s))
    if available:
        return np.where(string_array == '1')[0]
    else:
        return np.where(string_array != '1')[0]
    

def plot_percentage(x, cross_rewards, noughts_rewards, title='Title'):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel('Iteartions')
    plt.ylabel('Percentage')
    plt.grid(True)
    plt.plot(x, cross_rewards, label='Crosses')
    plt.plot(x, noughts_rewards, label='Noughts')
    plt.legend()
    plt.show()