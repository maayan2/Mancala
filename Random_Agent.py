import random
from graphics import *
from State import State

class Random_Agent():

    def __init__(self, player=1) -> None:
        self.palyer = player

    def get_action (self, event = None, state: State = None, train = None):
        if state.is_another_turn:
            return -1

        # if state.turn == 0:
        #     piece_per_allowed_pockets = state.board[0:6]
        # else:
        #     piece_per_allowed_pockets = state.board[7:13]
        # indexes_possible_pockets = [] #keeps indexes of possible pockets that does not have 0 pieces
        # for i in range(6):
        #     if piece_per_allowed_pockets[i] != 0:
        #         if state.turn == 0:
        #             indexes_possible_pockets.append(i)
        #         else:
        #             indexes_possible_pockets.append(i + 7)

        # action = random.choice(indexes_possible_pockets) #random value from the input list
        # return action 
        indexes_possible_pockets = []
        if state.turn == 0: #player A, range 0 - 5
            indexes_possible_pockets = list(np.where(state.board[0:6]!=0)[0])
               
        else: #player B, range 7 - 12
            indexes_possible_pockets = list(np.where(state.board[7:13]!=0)[0]+7)
        action = random.choice(indexes_possible_pockets) #random value from the input list
        return action     