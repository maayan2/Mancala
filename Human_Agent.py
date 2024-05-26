import pygame
from graphics import *
from State import State
import logging

class Human_Agent():

    def __init__(self, graphics, player, mancala, state):
        self.graphics = graphics
        self.player = player #determine hows turn it is
        self.mancala = mancala
        self.state = state

    # when player gets another turn, the opp action become: ( -1 ), 
    # and he basicly does nothing. by that, we "skip" his turn and return to the player to let him complete his turn.
    def get_action(self, events, state : State):
        if state.is_another_turn:
            return -1
        
        allowed_pockets = []
        if state.turn == 0:
            allowed_pockets = [0, 1, 2, 3, 4, 5]
        else:
            allowed_pockets = [7, 8, 9, 10, 11, 12]

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_presses = pygame.mouse.get_pressed()
                turn_start_pocket = self.graphics.is_in_pocket(pygame.mouse.get_pos(), allowed_pockets, state.board)#self.state
                if (turn_start_pocket != None and state.board[turn_start_pocket] == 0):
                    self.mancala.turn_title[4] = "Select a pocket with pieces!"
                    return None
                elif turn_start_pocket == None:
                    return None
                return turn_start_pocket