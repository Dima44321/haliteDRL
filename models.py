import torch
import torch.nn as nn
import numpy as np
from board_funcs import get_sample_board, board_to_np, augment_ship, augment_sy

#Constant params
nc = 3   #number channels
nf = 32  #number feature detectors
board_size = 21
ship_action_space_size = 5
shipyard_action_space_size = 2
agent_count = 4

#Network to control ships
class ShipNet(nn.Module):
    def __init__(self):
        super(ShipNet, self).__init__()
        
        #four convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 1, 0), # out 18x18
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True), 

            nn.Conv2d(nf, nf*2, 3, 1, 0), # out 16x16
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True), 

            nn.Conv2d(nf*2, nf*4, 4, 2, 1), # out 8x8
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*4, nf*2, 4, 2, 1), # out 4x4
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*2, nf, 4, 2, 1), # out 2x2
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        #single linear layer
        self.linear = nn.Sequential(
            nn.Linear(nf*4, ship_action_space_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        #forward pass through the network
        x = self.conv(x)
        x = x.view(-1, nf*4)
        return self.linear(x)

#Network to Control Shipyards
class ShipYardNet(nn.Module):
    def __init__(self):
        super(ShipYardNet, self).__init__()
        
        #four convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 1, 0), # out 18x18
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True), 

            nn.Conv2d(nf, nf*2, 3, 1, 0), # out 16x16
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True), 

            nn.Conv2d(nf*2, nf*4, 4, 2, 1), # out 8x8
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*4, nf*2, 4, 2, 1), # out 4x4
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*2, nf, 4, 2, 1), # out 2x2
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        #single linear layer
        self.linear = nn.Sequential(
            nn.Linear(nf*4, shipyard_action_space_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        #forward pass through the network
        x = self.conv(x)
        x = x.view(-1, nf*4)
        return self.linear(x)


#Coordinates Ships and Shipyards in order to produce actions given board
class Controller(): 
    def __init__(self, shipnet, synet, ships, sy):
        self.shipnet = shipnet
        self.synet = synet
        self.ships = ships
        self.sy = sy
    
    #return actions for ships and shipyards given a board state
    def get_actions(board):
        ship_actions = []
        sy_actions = []

        #step through ships and get action for each 
        for i in self.ships:
            board = augment_ship(board, i)
            ship_actions.append(self.shipnet(board))
        
        #step through shipyards and get action for each
        for i in self.sy:
            board = augment_sy(board, i)
            sy_actions.append(self.synet(board))

        return ship_actions, sy_actions
        