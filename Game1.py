import pygame
from graphics import *
from mancala import *
import time
from State import State
from Human_Agent import *
from Random_Agent import *
from MinMaxAgent import *
from AlphaBetaAgent import *
from DQN_Agent import *

# player A -> 0
# player B -> 1

def main ():    

    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Mancala')
    FPS = 60
    
    clock = pygame.time.Clock()
    graphics = Graphics(win)
    
    state = State()
    mancala = Mancala(state)
    state.init_Board()# reset the board to start state

    #### Agents:
    # agent1 = Human_Agent(graphics, player = 0, mancala = mancala, state = state)
    # agent1 = MinMaxAgent(graphics, player = 0,depth = 1, mancala = mancala)
    # agent1 = Random_Agent()
    # agent1 = DQN_Agent(player = 0,parametes_path=None, train=False, env = mancala)
    agent1 = AlphaBetaAgent(graphics, player = 0,depth=1, mancala = mancala, state = state)

    # checkpoint_path = "Data/checkpoint94000.pth"
    # checkpoint = torch.load(checkpoint_path)
    # agent1.DQN.load_state_dict(checkpoint['best_random_params'])

    # agent2 = Human_Agent(graphics, player = 1, mancala = mancala, state = state)
    # agent2 = Random_Agent()
    # agent2 = MinMaxAgent(graphics, player = 1,depth = 1, mancala = mancala)
    # agent2 = AlphaBetaAgent(graphics, player = 1,depth=5, mancala = mancala, state = state)
    agent2 = DQN_Agent(player = 1,parametes_path=None, train=False, env = mancala)

    checkpoint_path1 = "Data/checkpoint5016.pth"
    checkpoint1 = torch.load(checkpoint_path1)
    agent2.DQN.load_state_dict(checkpoint1['best_random_params'])

    agent = agent1

    mancala.turn == 0 #current player, player A starts the game
    turn_completed = False
    graphics.set_turn(mancala.turn_title)
    graphics.draw_board(state = mancala.state)

    game_actions = ''
    player_letter = 'A'
    mancala.turn_title[0] = 0

    run = True
    while(run):
        clock.tick(FPS)
        # pygame.time.delay(300)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                run = False

        action = agent.get_action(events, state)
        state.action = action

        if (action != None): #check if valid input -> if player chose one of his pockets
            if state.turn == 0:
                letter = 'A'
            else: 
                letter = 'B'
            game_actions = game_actions + letter + str(action) + ', ' 

            # make the move
            mancala.next_state(state=state, action=action)
            turn_completed = True
            graphics.set_turn(mancala.turn_title)

        if mancala.is_end_state(state):
            if isinstance(agent, Random_Agent):
                mancala.end_state(player_finisher=mancala.turn_title[0], state=state)
            else:
                mancala.end_state(player_finisher=mancala.turn_title[0], state=state)
            mancala.turn_title[4] = "                                                     "
            graphics.set_turn(mancala.turn_title)
        else:
            if turn_completed:
                if state.is_another_turn or action == -1: #player gets another turn
                    mancala.turn_title[4] = "Get another turn!                  "
                else:
                    mancala.turn_title[4] = "                                              "
                    if player_letter == 'A':
                        player_letter = 'B'
                    else:
                        player_letter = 'A'

                    mancala.turn_title[0] = state.turn

                if agent == agent1:
                    agent = agent2
                else: 
                    agent = agent1
                turn_completed = False            

        graphics.draw_board(state = mancala.state)
        pygame.display.update()

        if mancala.is_end_state(state):
            time.sleep(15)
            run = False

if __name__ == '__main__':
    main()