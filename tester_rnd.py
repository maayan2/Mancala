from Random_Agent import *
from mancala import *
from DQN_Agent import *
from AlphaBetaAgent import *
from MinMaxAgent import *

class tester_rnd:
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
                if self.env.has_won(env.state, 0):
                    player1_win += 1
                elif self.env.has_won(env.state, 1):
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
    # param_path = "Data/checkpoint01001.pth"
    state = State()
    state.init_Board()
    env = Mancala(state)
    
    # player1 = Random_Agent()
    player1 = AlphaBetaAgent(graphics=None, player = 0,depth=5, mancala = env, state = state)
    # player1 = MinMaxAgent(graphics = None, player = 0,depth = 5, mancala = env)
    # player1 = DQN_Agent(player = 0, env = env)
    # checkpoint_path = "Data/checkpoint01001.pth"
    # checkpoint = torch.load(checkpoint_path)
    # player1.DQN.load_state_dict(checkpoint['best_random_params'])
    # player1.DQN.load_state_dict(checkpoint['model_state_dict'])
    
    
    # player2 = DQN_Agent(player=1, env=env,train=False, parametes_path=param_path)
    player2 = Random_Agent()
    # player2 = MinMaxAgent(graphics = None, player = 1,depth = 7, mancala = env)
    # player2 = AlphaBetaAgent(graphics=None, player = 1,depth=3, mancala = env, state = state)
    test = tester_rnd(env,player1, player2)
    
    print(test.test(200))