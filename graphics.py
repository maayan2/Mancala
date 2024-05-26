import numpy as np
import pygame
import time
from mancala import *
import random


pygame.init()

WIDTH, HEIGHT = 1200, 460
ROWS, COLS = 2, 8
SQUARE_SIZE = 1
POCKET_WIDTH = 1
POCKET_HEIGHT = 3

LINE_WIDTH = 2
PADDING = SQUARE_SIZE //5

OBJECT_CENTER = [(190, 350), (350, 350), (510, 350), (670, 350), (830, 350), (990, 350), (1060, 130, 100, 280),                 
                 (990, 205), (830, 205), (670, 205), (510, 205), (350, 205), (190, 205), (20, 130, 100, 280)]

STORE_PIECES_POSITION = [(28,92), (71,34), (76,252), (39,123), (33,93), (38,65), (71,59), (53,52), (21,86), (57,27), (19,157), (22,204), (35,22),
                         (57,43), (76,133), (41,196), (15,47), (19,33), (56,36), (72,15), (17,250), (64,206), (24,225), (24,174), (60,246), (44,110), (26,156), (14,77),
                         (81,93), (24,219), (57,133), (18,163), (16,139), (19,182), (74,128), (55,193), (22,89), (37,63), (63,195), (78,13),
                         (37,81), (38,221), (37,217), (57,115), (25,82), (46,45), (72,155), (56,120)]

PIECE_RADIUS = 13

#color
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
LIGHTGRAY = (211,211,211)
GREEN = (0, 128, 0)
ANTIQUEWHITE = (250,235,215)
CADETBLUE = (95,158,160)
CARROT = (237,145,33)
DARKOCHID = (153,50,204)
DEEPSKYBLUE	= (0,154,205)
MOCCASIN = (255,228,181)
CORNSLIK2 = (238,232,205)
CADETBLUE2 = (142,229,238)

class Graphics:
    def __init__(self, win):
        self.win = win
        self.turn_title = None
        self.board_objects = [None] * 14
        self.list_players = ["Player A", "Player B", "Both"] 
        self.color_list = [RED, BLUE, GREEN, ANTIQUEWHITE, LIGHTGRAY, CADETBLUE, CARROT, DARKOCHID, DEEPSKYBLUE]

    def set_turn(self, turn):
        self.turn_title = turn

    def draw_board(self, state):
        board = state.board
        pygame.draw.rect(self.win, BLACK, (0, 1, POCKET_WIDTH, POCKET_HEIGHT))
        for i in range(6):
            self.draw_pocket(i, board=board)
            self.draw_pocket(i + 7, board=board)
        self.draw_store(6, board=board)
        self.draw_store(13, board=board)
        self.draw_turn_title(state)

    def draw_pocket(self, index, color = WHITE, board=None):
        #pocket index is 0-5, 7-12
        if index == 13 or index == 6:
            return

        radius = 50
        self.board_objects[index] = pygame.draw.circle(self.win, color, OBJECT_CENTER[index], radius)
        self.draw_pieces(index, board)

    def draw_store(self, index, board):  
        #store index is 6 or 13
        if index != 13 and index != 6:
            return
        # Drawing Rectangle
        self.board_objects[index] = pygame.draw.rect(self.win, WHITE, pygame.Rect(OBJECT_CENTER[index]), 0, 10)
        self.draw_pieces(index, board)

    def draw_pieces(self, index, board):
        if index == 6 or index == 13:
            self.draw_pieces_store(index, board)
        else:
            self.draw_pieces_pocket(OBJECT_CENTER[index], index, board)

    def draw_pieces_store(self, index, board):
        num_pieces = board[index] #.board_game

        for i in range(num_pieces):
            piece_center = (STORE_PIECES_POSITION[i][0] + OBJECT_CENTER[index][0],
                            STORE_PIECES_POSITION[i][1] + OBJECT_CENTER[index][1])
            pygame.draw.circle(self.win, self.color_list[i % len(self.color_list)], piece_center, PIECE_RADIUS)
   
        font = pygame.font.SysFont('arial', 24)
        if index == 13:
            display_str = "Player B (" + str(board[index]) + ")   " #.board_game
            text = font.render(display_str, True, MOCCASIN, BLACK)
            self.win.blit(text, (20, 80))
        if index == 6:
            display_str = "Player A (" + str(board[index]) + ")   " #.board_game
            text = font.render(display_str, True, MOCCASIN, BLACK)
            self.win.blit(text, (1060, 80))

    def draw_pieces_pocket(self, pocket_center, index, board):
        num_pieces = board[index] #.board_game
        pieces_list = []
        if num_pieces == 0:
            pieces_list = []
        if num_pieces == 1:
            pieces_list = [(-13, -13)]
        if num_pieces == 2:
            pieces_list = [(-13, -13), (11, -16)]
        if num_pieces == 3:
            pieces_list = [(-13, -13), (11, -16), (-7, 14)]
        if num_pieces == 4:
            pieces_list = [(-13, -13), (11, -16), (-7, 14), (12, 15)]
        if num_pieces == 5:
            pieces_list = [(-13, -13), (11, -16), (-7, 14), (12, 15), (5, 25)]
        if num_pieces == 6:
            pieces_list = [(-13, -13), (11, -16), (-7, 14), (12, 15), (5, 25), (-19, -24)]
        if num_pieces == 7:
            pieces_list = [(-13, -13), (11, -16), (-7, 14), (12, 15), (5, 25), (-19, -24), (12,-9)]
        if num_pieces >= 8:
            pieces_list = [(-13, -13), (11, -16), (-7, 14), (12, 15), (5, 25), (-19, -24), (12,-9), (-9, -9), (-10,-11), (-9,14)]

        for i in range(len(pieces_list)):
            piece_center = (pocket_center[0] + pieces_list[i][0], pocket_center[1] + pieces_list[i][1])
            pygame.draw.circle(self.win, self.get_color(i), piece_center, PIECE_RADIUS)

        font = pygame.font.SysFont('arial', 24)
        if index < 6:
            display_str = "A" + str(index + 1) + "(" + str(board[index]) + ")   " #.board_game
            text = font.render(display_str, True, MOCCASIN, BLACK)
            self.win.blit(text, (pocket_center[0] - 19, pocket_center[1] + 60))
        if index > 6 and index < 13:
            display_str = "B" + str(index - 6) + "(" + str(board[index]) + ")   " #.board_game
            text = font.render(display_str, True, MOCCASIN, BLACK)
            self.win.blit(text, (pocket_center[0] - 19, pocket_center[1] - 87))

    def get_color(self, index=None):
        
        if index == None:
            return self.color_list[random.randint(0, len(self.color_list) - 1)]
        else:
            return self.color_list[index % len(self.color_list)]
    
    def is_in_pocket(self, position, allowed_pockets, board):
        index_of_pocket = None
        for i in allowed_pockets:
            if self.board_objects[i].collidepoint(position):
                self.draw_pocket(i, CORNSLIK2, board)
                index_of_pocket = i
            else:
                self.draw_pocket(i, WHITE, board)
        return index_of_pocket
    
    def draw_turn_title(self, state):
        font = pygame.font.SysFont('arial', 30)
        if self.turn_title[3] == None:
            display_str = "Current turn: " + self.list_players[self.turn_title[0]] # print current turn
            text = font.render(display_str, True, CADETBLUE2, BLACK)
            self.win.blit(text, (460, 50))

            display_str = self.turn_title[4] # get another turn
            text = font.render(display_str, True, RED, BLACK)
            self.win.blit(text, (800, 50))
    
        else:
            display_str = "Game over!   "# end of game
            text = font.render(display_str, True, RED, BLACK)
            self.win.blit(text, (200, 50))
            if self.turn_title[3] != 2:
                display_str = "The winner is " + self.list_players[self.turn_title[3]] + ", with " + str(self.turn_title[self.turn_title[3] + 1]) + " points                                  " # print winner and his score
            else:
                display_str = "The winner is " + self.list_players[self.turn_title[3]] + ", with " + str(self.turn_title[1]) + " points each                        "# print winner and his score
            text = font.render(display_str, True, RED, BLACK)
            self.win.blit(text, (360, 50))