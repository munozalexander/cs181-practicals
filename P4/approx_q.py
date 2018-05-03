# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import time
import pickle

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.first_tick = True
        self.screen_width = 600
        self.screen_height = 400
        self.max_vel = 60
        self.gamma = 0.5
        self.eta = 0.5
        self.step = 0
        self.epsilon_decay = 0.999
        self.w = np.zeros(6)

    def Q(self, g, x, y, v, t, a):
        return np.dot([g,x,y,v,t,a], self.w)

    def epsilon(self):
        return self.epsilon_decay ** self.step

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.first_tick = True

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        new_state  = state

        if self.first_tick:
            if self.last_state is not None:
                self.high_grav = int(int(state['monkey']['vel'] - self.last_state['monkey']['vel']) == -4)
                self.first_tick = False
                self.step += 1
            else:
                self.last_state = new_state
                self.last_action = 0
                return self.last_action

        monkey_x = new_state['tree']['dist']
        monkey_y = new_state['monkey']['bot']
        monkey_v = new_state['monkey']['vel']
        tree_y = new_state['tree']['bot']

        monkey_x_old = self.last_state['tree']['dist']
        monkey_y_old = self.last_state['monkey']['bot']
        monkey_v_old = self.last_state['monkey']['vel']
        tree_y_old = self.last_state['tree']['bot']

        prev_q = self.Q(self.high_grav, monkey_x_old, monkey_y_old, monkey_v_old, tree_y_old, self.last_action)
        grad_q = prev_q - (self.last_reward + self.gamma * np.max([self.Q(self.high_grav, monkey_x, monkey_y, monkey_v, tree_y, 0),self.Q(self.high_grav, monkey_x, monkey_y, monkey_v, tree_y, 1)]))
        self.w -= self.eta*grad_q*np.array([self.high_grav, monkey_x_old, monkey_y_old, monkey_v_old, tree_y_old, self.last_action])

        if np.random.random() < self.epsilon():
            new_action = np.random.choice(2)
        else:
            new_action = np.argmax([self.Q(self.high_grav, monkey_x, monkey_y, monkey_v, tree_y, 0),self.Q(self.high_grav, monkey_x, monkey_y, monkey_v, tree_y, 1)])

        self.last_action = new_action
        self.last_state  = new_state
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    timestamp = int(time.time())
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)
        print('epoch: ' + str(ii) + ', score: ' + str(swing.score) + ', gravity: ' + str(swing.gravity) + ', running_avg: ' + str(np.average(hist[-10:])))

        results = {
            'gamma': learner.gamma,
            'eta': learner.eta,
            'epsilon_decay': learner.epsilon_decay,
            'hist': hist
        }

        with open ('results/results_approx_' + str(timestamp) + '.p', 'wb') as f:
            pickle.dump(results, f)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    agent = Learner()
    hist = []
    run_games(agent, hist, 1000, 1)
    print(hist)
    np.save('hist',np.array(hist))
