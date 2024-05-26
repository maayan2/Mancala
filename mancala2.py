import pygame
from graphics import *
import random
import logging
from State import State
import numpy as np


'''
Game Play:
1. The game begins with one player picking up all of the pieces in any one of the pockets on his/her side.
2. Moving counter-clockwise, the player deposits one of the stones in each pocket until the stones run out.
3. If you run into your own Mancala (store), deposit one piece in it. If you run into your opponent's Mancala, skip it and continue moving to the next pocket.
4. If the last piece you drop is in your own Mancala, you take another turn.
5. If the last piece you drop is in an empty pocket on your side, you capture that piece and any pieces in the pocket directly opposite.
6. Always place all captured pieces in your Mancala (store).
7. The game ends when all six pockets on one side of the Mancala board are empty.
8. The player who still has pieces on his/her side of the board when the game ends captures all of those pieces.
9. Count all the pieces in each Mancala. The winner is the player with the most pieces.
'''

#environment#
class Mancala:
	#  [  13  ] [12, 11, 10,  9,  8,  7] [storeB]
	#  [storeA] [ 0,  1,  2,  3,  4,  5] [   6  ]

    def __init__(self, state: State = None) -> None:      
        if state is None:
            self.turn = 0
            self.turn_title = [0, None, None, None, None] # [current turn, player a score, player b score, winner, additional info]
        else:    
            self.turn = state.turn 
            self.state = state
            self.turn_title = [self.state.turn, None, None, None, None] # [current turn, player a score, player b score, winner, additional info]
        self.winner = ""
        self.get_another_turn = False

    def get_init_state(self):
        board = np.array ([4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]) #start state !!!!! bug - list->numpy
        return State (board, turn=0, is_another_turn=False)

    # הפעולה מקבלת אינדקס תקין התחלתי, מבצעת את התקדמות הגולות בתור, מעדכנת את הלוח ולמעשה נגמר התור 
    def next_state(self, state: State, action, is_DQN = False):

        state.scored = 0 # counter of how many were scored in player's big pocket during one move

        if action == -1:
            state.switch_player()
            state.is_another_turn = False
            return
        
        state.scored = 0 # counter of how many were scored in player's big pocket during one move

        allowed_pockets = []
        if state.turn == 0:
            allowed_pockets = [0, 1, 2, 3, 4, 5]
        else:
            allowed_pockets = [7, 8, 9, 10, 11, 12]
        num_pieces = state.board[action]
        if state.turn == 0:
            other_player_store_index = 13
            my_player_store_index = 6
        else:
            other_player_store_index = 6
            my_player_store_index = 13

        if num_pieces > 0: #player must pick a pocket that has at least one piece
            state.board[action] = 0 # player pick up all pieces from current pocket
            i = 1
            #for i in range(1, num_pieces + 1):
            while(num_pieces > 0):      
                place = (action + i) % 14 # pocket for the next piece  
                if place != other_player_store_index:
                    state.board[place] = state.board[place] + 1
                    num_pieces = num_pieces - 1

                if place == my_player_store_index:
                    state.scored += 1

                if num_pieces == 0:
                    state.is_another_turn = False
                    self.get_another_turn = False
                    if place == my_player_store_index:
                        state.is_another_turn = True
                        self.get_another_turn = True #last piece is in player's store -> gets another turn

                    ## 5. If the last piece you drop is in an empty pocket on your side, you capture that piece and any pieces in the pocket directly opposite.
                    if state.board[place] == 1 and place != 6 and place != 13 and place in allowed_pockets:
                        state.board[my_player_store_index] = state.board[my_player_store_index] + state.board[12 - place]
                        state.scored += state.board[12 - place]
                        state.board[12 - place] = 0
                i = i + 1
        if is_DQN:
            if self.is_end_state(state):
                self.end_state(player_finisher=state.turn, state=state)

        state.switch_player()
        
    # get state return boolean            
    def is_end_state(self, state):
        #check if end state
        is_end = True
        for i in range(6):
              if state.board[i] > 0:
                is_end = False
                break
        
        if is_end == False:
            is_end = True
            for i in range(7,13):
                if state.board[i] > 0:
                    is_end = False
                    break

        return is_end
    
    #calculate scores of every player and find the winner
    # get state and calc the state score - return score (player 1 - player 2)
    def end_state(self, player_finisher, state): 
        if player_finisher == 0:
            my_range = (0, 6)
            other_range = (7, 12)
            my_player_store_index = 6
            other_player = 1
        else:
            my_range = (7, 13)
            other_range = (0, 5)
            my_player_store_index = 13
            other_player = 0
        
        #capture all pieces from losing player
        for i in range(other_range[0], other_range[1] + 1):
            state.board[my_player_store_index] += state.board[i]
            state.scored += state.board[i]
            state.board[i] = 0

        #winner
        if state.board[my_player_store_index] > state.board[other_range[1] + 1]:
            self.winner = player_finisher
        elif state.board[my_player_store_index] < state.board[other_range[1] + 1]:
            self.winner = other_player
        else:
            self.winner = 2 # both players have the same score

        if player_finisher == 0:
            self.turn_title = ["       ", state.board[my_player_store_index], state.board[other_range[1] + 1], self.winner, None]
        else:
            self.turn_title = ["       ", state.board[other_range[1] + 1], state.board[my_player_store_index], self.winner, None]

        # self.graphics.set_turn(self.turn_title)

    def all_legal_action(self, state: State):
        if state.is_another_turn:
            return [-1]
        actions = []
        # if state.turn == 0: #player A, range 0 - 5
        #     actions = [0,1,2,3,4,5]
        # else: #player B, range 7 - 12
        #     actions = [7,8,9,10,11,12]
        # for action in range(6):
        #     if state.board[action] == 0:
        #         if state.turn == 0:
        #             actions.remove(action)
        #         else:
        #             actions.remove(action + 7)
        # return actions
        if state.turn == 0: #player A, range 0 - 5
            actions = list(np.where(state.board[0:6]!=0)[0])
               
        else: #player B, range 7 - 12
            actions = list(np.where(state.board[7:13]!=0)[0]+7)
        return actions

    #Returns whether or not the given player has won
    def has_won(self, state, turn):
        if self.is_end_state(state):
            self.end_state(turn, state) 
            if self.winner == turn:
                return True
            else:
                return False
            
    def has_won2(self, state, turn):
        if self.is_end_state(state):
            if self.winner == turn:
                return True
            else:
                return False
        
    def get_next_state(self, action, state: State = None, is_DQN = False):
        next_move = state.copy()
        self.next_state(state = next_move, action = action, is_DQN = is_DQN) 

        return next_move
    
    def evaluate (self, state:State, player): 
        player = player
        opponent = (player * -1) + 1
        if self.has_won2(state, player ):    
            return 100.0
        if self.has_won2(state, opponent): 
            return -100.0
        pocketA, pocketB = state.board[6], state.board[13]
        if player == 0:
            value = pocketA - pocketB
        else: 
            value = pocketB - pocketA
        if state.is_another_turn and state.action != -1: 
            value += 5
        if state.is_another_turn and state.action == -1: 
            value -= 5
        return value
    
    def reward (self, state : State, is_another_turn : bool) -> tuple:
        # if action:
        #     next_state = self.get_next_state(action, state)
        # else:
        # next_state = state
        # if (self.is_end_state(next_state)):
        #     # self.end_state(next_state.turn, next_state)
        #     score = state.board[6] - state.board[13]
        #     if score > 0:
        #         return 1, True
        #     if score < 0:
        #         return -1, True
        #     return 0, True
        # return 0, False
        
        next_state = state
        if (self.is_end_state(next_state)):
            # self.end_state(next_state.turn, next_state)
            score = state.board[6] - state.board[13]
            if score > 0:
                return 2, True
            if score < 0:
                return -2, True
            return 0, True
        if next_state.turn == 1:
            return next_state.scored*0.1, False
        else:
            return -next_state.scored*0.1, False  
     
    def restart (self):   
        self.turn = 0
        self.turn_title = [self.state.turn, None, None, None, None] # [current turn, player a score, player b score, winner, additional info]
        self.winner = ""
        self.get_another_turn = False