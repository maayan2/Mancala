import numpy as np
import torch

class State:
    def __init__(self, board = None, turn = 0, is_another_turn = False, action = None, scored = 0):
        self.board = board
        self.turn = turn
        self.is_another_turn = is_another_turn      # bool -> gets another turn or not
        self.action = action                        # same as turn_start_pocket (action=99 is fectivy)
        self.scored = scored                        # counter of how many were scored in player's big pocket during one move

    def init_Board (self):
        self.board = np.array([4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]) #start state bug change to numpy
        self.turn = 0
        self.is_another_turn = False
        self.action = None

    def switch_player (self):
        if self.turn == 0:
            self.turn = 1
        else:
            self.turn = 0
  
    def all_legal_action(self):
        if self.is_another_turn:
            return -1
        actions = []
        # if self.turn == 0: #player A, range 0 - 5
        #     actions = [0,1,2,3,4,5]
        # else: #player B, range 7 - 12
        #     actions = [7,8,9,10,11,12]
        # for action in range(6):
        #     if self.board[action] == 0:
        #         if self.turn == 0:
        #             actions.remove(action)
        #         else:
        #             actions.remove(action + 7)
        # return actions
        if self.turn == 0: #player A, range 0 - 5
            actions = list(np.where(self.board[0:6]!=0)[0])
               
        else: #player B, range 7 - 12
            actions = list(np.where(self.board[7:13]!=0)[0]+7)
        return actions

    def __eq__(self, other) ->bool:
        b1 = np.equal(self.board, other.board).all()
        b2 = self.turn == other.turn
        b3 = self.is_another_turn == other.is_another_turn
        return b1 and b2 and b3

    def __hash__(self) -> int:
        return hash(repr(self.board) + repr(self.turn) + repr(self.is_another_turn))

    def copy (self):
        newBoard = np.copy(self.board)
        return State(board=newBoard, turn=self.turn, is_another_turn = self.is_another_turn, action = self.action)      
    
    def toTensor (self, device = torch.device('cpu')) -> tuple:
        board_tensor = torch.tensor(self.board)
               # board_tensor = torch.tensor([self.board], dtype=torch.float32, device=device)
        turn = torch.tensor([self.turn])
        is_another_turn = torch.tensor([self.is_another_turn])
        # action = torch.tensor([self.action])
        tensor = torch.cat((board_tensor, turn, is_another_turn)).type(torch.float32)
        
        return tensor
    
    def reverse (self):
        reversed = self.copy()
        if self.turn == 0:
            reversed.turn = 1
        else:
            reversed.turn = 0
        reversed.board[0:7], reversed.board[7:14] = self.board[7:14], self.board[0:7]
        if self.action != None:
            reversed.action = (self.action + 7) % 14
        else:
            reversed.action = None
        return reversed
    
    @staticmethod
    def tensorToState (state_tensor, player):
        board = state_tensor.cpu().numpy()[0:14]
        playerr = int(state_tensor[14].item())
        is_another_turn = bool(state_tensor[15])
        # action = state_tensor[16]
        return State(board, turn=playerr, is_another_turn=is_another_turn)