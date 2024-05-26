from mancala import *
from State import State
import logging

MAXSCORE = 1000

class AlphaBetaAgent():
    def __init__(self, graphics, player, depth = 2, mancala : Mancala = None, state: State = None, train = None):
        self.graphics = graphics
        self.player = player #determine hows turn it is
        if self.player == 0:
            self.opponent = 1
        else:
            self.opponent = 0
        self.mancala : Mancala = mancala
        self.state = state
        self.depth = depth
        
    def get_action(self, event = None, state : State = None, train = None):
        visited = set()
        value, bestAction = self.minMax(state, visited)
        return bestAction
        
    #Returns the score for this player given the state of the board
    def evaluate (self, state:State, player): 
        player = player
        opponent = (player * -1) + 1
        if self.mancala.has_won2(state, player ):    
            return 100.0
        if self.mancala.has_won2(state, opponent): 
            return -100.0
        pocketA, pocketB = state.board[6], state.board[13]
        # value = (pocketB - pocketA) * (self.player*2 - 1)
        if player == 0:
            value = pocketA - pocketB
        else: 
            value = pocketB - pocketA
        if state.is_another_turn and state.action != -1: 
            value += 5
        if state.is_another_turn and state.action == -1: 
            value -= 5
        return value

    def minMax(self, state : State, visited:set):
        depth = 0
        alpha = -MAXSCORE
        beta = MAXSCORE
        return self.max_value(state, visited, depth, alpha, beta)

    def max_value (self, state: State, visited:set, depth, alpha, beta):
        
        # stop state
        if depth == self.depth or self.mancala.is_end_state(state):
            value = self.mancala.evaluate(state, self.player)
            return value, None #self.mancala.turn_start_pocket
                
        # start recursion
        value = -MAXSCORE
        bestAction = []
        legal_actions = self.mancala.all_legal_action(state)
        for action in legal_actions:
            newState = self.mancala.get_next_state(action, state)     
            if newState  in visited:
                continue
            visited.add(newState)
            newValue, newAction = self.min_value(newState, visited,  depth + 1, alpha, beta)
            if newValue > value:
                value = newValue
                bestAction = action
                alpha = max(alpha, value)
            if value >= beta:
                return value, bestAction
        return value, bestAction 

    def min_value (self, state, visited:set, depth, alpha, beta):
         
        # stop state
        if depth == self.depth or self.mancala.is_end_state(state):
            value = self.mancala.evaluate(state, self.player)
            return value, None #self.mancala.turn_start_pocket
                
        # start recursion
        value = MAXSCORE
        bestAction = []
        legal_actions = self.mancala.all_legal_action(state)
        for action in legal_actions:
            newState = self.mancala.get_next_state(action, state)     
            if newState  in visited:
                continue
            visited.add(newState)
            newValue, newAction = self.max_value(newState, visited,  depth + 1, alpha, beta)
            if newValue < value:
                value = newValue
                bestAction = action
                beta = min(beta, value)
            if value <= alpha:
                return value, bestAction
        return value, bestAction 