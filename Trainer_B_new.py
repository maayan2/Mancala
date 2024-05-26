from mancala import *
from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Random_Agent import Random_Agent
import torch
from Tester import Tester
import os
import wandb
#region ###### params ############
epochs = 2000000
start_epoch = 0
C = 30
learning_rate = 0.001
batch_size = 64
MIN_Buffer = 4000
# epsilon Greedy
epsilon_start = 1
epsilon_final = 0.001
epsiln_decay = 300
#endregion

def main ():
    start_epoch = 0
    state_1 = State()
    state_1.init_Board()# reset the board to start state
    env = Mancala(state_1)
    
    player1 = DQN_Agent(player=1, env=env,parametes_path=None)
    player_hat = DQN_Agent(player=1, env=env, train=False)
    Q = player1.DQN
    Q_hat = Q.copy()
    player_hat.DQN = Q_hat
    
    player2 = Random_Agent()   
    buffer = ReplayBuffer(path=None) # None
    
    results = [] #results_file['results'] # []
    avgLosses = [] #results_file['avglosses']     #[]
    avgLoss = 0 #avgLosses[-1] #0
    loss = 0
    res = 0
    best_res = -200
    loss_count = 0
    
    # random_results = [] #torch.load(random_results_path)   # []
    best_random = 0
    best_random_params = None

    # init optimizer
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,10000*30, gamma=0.50)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[30*50000, 30*100000, 30*250000, 30*500000], gamma=0.5)
    
    #region ######## checkpoint Load ############
    file_num = 502
    checkpoint_path = f"Data/checkpoint{file_num}.pth"
    buffer_path = f"Data/buffer{file_num}.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']+1
        player1.DQN.load_state_dict(checkpoint['model_state_dict'])
        player_hat.DQN.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        buffer = torch.load(buffer_path)
        results = checkpoint['results']
        avgLosses = checkpoint['avgLosses']
        best_res = checkpoint['best_res']
        best_random_params = checkpoint['best_random_params']
        best_random = checkpoint['best_random']
    player1.DQN.train()
    player_hat.DQN.eval()
    tester = Tester(player1=player1, player2=Random_Agent(), env=env)
    #endregion
    #region ################ Wandb.init #####################
    
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"Mancala{file_num}",
        resume=False, 
        id='Mancala',
        # track hyperparameters and run metadata
        config={
        "name": "Mancala",
        "checkpoint": checkpoint_path,
        "learning_rate": learning_rate,
        "Model": str(player1.DQN),
        "Schedule": f'{str(scheduler.step_size)} gamma={str(scheduler.gamma)}',
        "epochs": epochs,
        "start_epoch": start_epoch,
        "decay": epsiln_decay,
        "gamma": 0.99,
        "batch_size": batch_size, 
        "C": C,
        'player': 'player 1'
        }
    )
    #endregion
    for epoch in range(start_epoch, epochs):
        print(f'epoch = {epoch}', end='\r')
        state_1.init_Board()
        step = 0
        while not env.is_end_state(state_1):
            # Sample Environement
            step += 1
            # print(step, end="\r")
            action_1 = player1.get_action(events=None, state=state_1, epoch=epoch)
            is_another_turn = state_1.is_another_turn
            after_state_1 = env.get_next_state(state=state_1, action=action_1, is_DQN = True)
            reward_1, end_of_game_1 = env.reward(state=after_state_1, is_another_turn = is_another_turn)
            reward_1 *= -1
            # print(state_1.board.reshape(2,7), action_1, 'reward=',reward_1, 'scored=', after_state_1.scored)
            if end_of_game_1:
                res += reward_1
                #dif reward#
                # if reward_1 > 0:
                #     res += 1
                # elif reward_1 < 0:
                #     res -=1
                
                buffer.push(state_1, action_1, reward_1, after_state_1, True)
                # print(after_state_1.board.reshape(2,7), -1,'reward', 0)
                break

            state_2 = after_state_1
            action_2 = player2.get_action(state=state_2)
            is_another_turn = state_2.is_another_turn
            after_state_2 = env.get_next_state(state=state_2, action=action_2, is_DQN = True)
            reward_2, end_of_game_2 = env.reward(state=after_state_2, is_another_turn = is_another_turn)
            reward_2 *= -1
            reward_2 = reward_1 + reward_2
            # print(state_2.board.reshape(2,7), action_2, 'reward', reward_2, 'scored', after_state_2.scored)
            if end_of_game_2:
                # print(after_state_2.board.reshape(2,7), -1, 'reward',0)
                res += reward_2
                # dif reward #
                # if reward_2 > 0:
                #     res += 1
                # elif reward_2 < 0:
                #     res -=1
            buffer.push(state_1, action_1, reward_2, after_state_2, end_of_game_2)
            
            state_1 = after_state_2
            if len(buffer) < MIN_Buffer:
                continue
            
            # Train NN
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = Q(states, actions)
            next_actions = player_hat.get_Actions(next_states, dones) 
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states, next_actions) #todo: use the values calculated in get_Actions

            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            scheduler.step()
            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001 
            
        if epoch % C == 0:
            Q_hat.load_state_dict(Q.state_dict())

        if (epoch+1) % 100 == 0:
            avgLosses.append(avgLoss)
            results.append(res)
            best_res = max(best_res, res)
            wandb.log ({
                "result": res,
                "avgLoss": avgLoss,
            })
            res = 0
        
        #### Finding best model ###########
        if (epoch+1) % 100 == 0:
            test = tester(100)
            test_score = test[1]-test[0]
            if best_random < test_score:
                best_random = test_score
                best_random_params = player1.DQN.state_dict().copy()
            print(test)

        #### Saving checkpoint ###########
        if epoch % 500 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': player1.DQN.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'results': results,
                'avgLosses':avgLosses,
                'best_res': best_res,
                'best_random_params': best_random_params,
                'best_random': best_random
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(buffer, buffer_path)
    
        print (f'epoch={epoch} step={step} loss={loss:.5f} avgloss={avgLoss:.5f}', end=" ")
        print (f'learning rate={scheduler.get_last_lr()[0]} path={checkpoint_path} res= {res} best_res = {best_res}')

    torch.save(checkpoint, checkpoint_path)
    torch.save(buffer, buffer_path)
    

if __name__ == '__main__':
    main()