# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

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
        self.n_bins = 5
        self.eta = .1
        self.Q = np.zeros((self.n_bins, self.n_bins, self.n_bins, self.n_bins, 2))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def discretize(self, val, max_val):
        bin_width = max_val / self.n_bins
        bins = [bin_width*i for i in range(1,self.n_bins+1)]
        loc = [val<=a for a in bins].index(True)
        return loc

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        new_state  = state

        if self.first_tick:
            if self.last_state is not None:
                self.gravity = state['monkey']['vel'] - self.last_state['monkey']['vel']
                self.first_tick = False
            self.last_state = new_state
            self.last_action = 0
            return self.last_action

        monkey_x = self.discretize(new_state['tree']['dist'], self.screen_width)
        monkey_y = self.discretize(new_state['monkey']['bot'], self.screen_height)
        monkey_v = self.discretize(new_state['monkey']['vel'], self.max_vel)
        tree_y = self.discretize(new_state['tree']['bot'], self.screen_height)

        new_action = np.argmax(self.Q[monkey_x, monkey_y, monkey_v, tree_y, :])
        self.last_action = new_action
        self.last_state  = new_state
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward
        monkey_x = self.discretize(self.last_state['tree']['dist'], self.screen_width)
        monkey_y = self.discretize(self.last_state['monkey']['bot'], self.screen_height)
        monkey_v = self.discretize(self.last_state['monkey']['vel'], self.max_vel)
        tree_y = self.discretize(self.last_state['tree']['bot'], self.screen_height)
        q_curr = self.Q[monkey_x, monkey_y, monkey_v, tree_y, self.last_action]
        q_fut = self.Q[monkey_x, monkey_y, monkey_v, tree_y]


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
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

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 20, 10)

	# Save history.
	np.save('hist',np.array(hist))
