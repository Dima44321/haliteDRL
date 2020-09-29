import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *


def get_sample_board(board_size, agent_count):
    environment = make("halite", configuration={
                       "size": board_size, "startingHalite": 1000})
    environment.reset(agent_count)
    state = environment.state[0]
    board = Board(state.observation, environment.configuration)
    return board


def board_to_np(board, board_dim, board_size=21, max_cell_halite=9, agent_count=4):
    '''Converts board to three channel array by first converting to string and sorting with ASCII
    board_dim specifies the number of channels'''

    # convert board to string and clean up
    board_np = str(board).replace(' ', "")
    board_np = board_np.replace('|\n', "")

    # tokenize board to insert into np array
    board_np = np.array((board_np.split("|"))[1:])

    # create placeholder array that will hold board
    board_arr = np.zeros((board_dim, board_size**2))

    # iterate through cells and fill in placeholder array
    for i in range(board_size**2):
        cell = board_np[i]
        for j in cell:
            if j >= '0' and j <= '9':  # using ASCII to determine character
                # normalize between 0-1
                board_arr[0, i] = int(j)/max_cell_halite
            elif j >= 'A' and j <= 'Z':
                board_arr[1, i] = (ord(j)-64)/agent_count
            elif j >= 'a' and j <= 'z':
                board_arr[2, i] = (ord(j)-96)/agent_count
    board_arr = board_arr.reshape(board_dim, board_size, board_size)

    # return np array of size 3 x n x n
    return board_arr
