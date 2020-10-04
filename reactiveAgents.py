# reactiveAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import numpy as np
from learning import training, get_data

class NaiveAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        sense = state.getPacmanSensor()
        if sense[7]:
            return Directions.STOP
        else:
            return Directions.WEST

class PSAgent(Agent):
    "An agent that follows the boundary using production system."

    def getAction(self, state):
        ''' Your code goes here! '''
        sense = state.getPacmanSensor()
        if sense[1] and sense[2] and not sense[3]:
            return Directions.EAST
        elif sense[3] and sense[4] and not sense[5]:
            return Directions.SOUTH
        elif sense[5] and sense[6] and not sense[7]:
            return Directions.WEST
        else:
            return Directions.NORTH
        return Directions.NORTH

class ECAgent(Agent):
    "An agent that follows the boundary using error-correction."

    def getAction(self, state):
        sense = np.append(state.getPacmanSensor(),1)
        east, east_r = get_data("east.csv")
        west, west_r = get_data("west.csv")
        south, south_r = get_data("south.csv")
        north, north_r = get_data("north.csv")
        east_d = np.dot(training(east, east_r,1),sense)>=0
        west_d = np.dot(training(west, west_r,1),sense)>=0
        south_d = np.dot(training(south, south_r,1),sense)>=0
        north_d = np.dot(training(north, north_r,1),sense)>=0

        if north_d:
            return Directions.NORTH
        elif east_d:
            return Directions.EAST
        elif south_d:
            return Directions.SOUTH
        elif west_d:
            return Directions.WEST
        else:
            return Directions.NORTH

