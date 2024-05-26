from Random_Agent import *
from mancala2 import *
from DQN_Agent import *
from AlphaBetaAgent import *
from MinMaxAgent import *

class Tester:
    def __init__(self, env : Mancala, player1, player2) -> None:
        self.env = env
        self.player1 = player1
        self.player2 = player2

    def test (self, games_num):
        env = self.env 
        player = self.player1
        player1_win = 0
        player2_win = 0
        games = 0
        while games < games_num:
            print (games, end='\r')
            action = player.get_action(state=env.state, train = False)
            env.next_state(state=env.state, action=action)
            player = self.switchPlayers(player)
            if env.is_end_state(env.state):
                score, _ = self.env.reward(env.state, False)
                if score > 0:
                    player1_win += 1
                elif score < 0:
                    player2_win += 1
                env.restart()
                env.state = env.get_init_state()
                games += 1
                player = self.player1
        return player1_win, player2_win        

    def switchPlayers(self, player):
        if player == self.player1:
            return self.player2
        else:
            return self.player1

    def __call__(self, games_num):
        return self.test(games_num)

if __name__ == '__main__':
    # param_path = "Data/checkpoint94000.pth"
    state = State()
    state.init_Board()
    env = Mancala(state)
    
    player1 = Random_Agent()
    # player1 = DQN_Agent(player = 0, env = env)
    # checkpoint_path1 = "Data/checkpoint70.pth"
    # checkpoint1 = torch.load(checkpoint_path1)
    # player1.DQN.load_state_dict(checkpoint1['best_random_params'])
    # player1.DQN.load_state_dict(checkpoint1['model_state_dict'])
    # player1 = MinMaxAgent(graphics = None, player = 0,depth = 3, mancala = env)


    player2 = DQN_Agent(player=1, env=env)
    checkpoint_path2 = "Data/checkpoint5015.pth"
    checkpoint2 = torch.load(checkpoint_path2)
    # player2.DQN.load_state_dict(checkpoint2['best_random_params'])
    player2.DQN.load_state_dict(checkpoint2['model_state_dict'])
    # player2 = Random_Agent()
    # player2 = AlphaBetaAgent(graphics=None, player = 1,depth=3, mancala = env, state = state)
    test = Tester(env,player1, player2)
    
    print(test.test(100))