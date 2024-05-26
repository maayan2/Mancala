from mancala2 import *
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
epsilon_final = 0.01
epsiln_decay = 300
#endregion

def main ():
    start_epoch = 0
    state_1 = State()
    state_1.init_Board()# reset the board to start state
    env = Mancala(state_1)
    
    player1 = DQN_Agent(player=0, env=env,parametes_path=None)
    player1_hat = DQN_Agent(player=0, env=env, train=False)
    Q1 = player1.DQN
    Q1_hat = Q1.copy()
    player1_hat.DQN = Q1_hat
    
    player2 = DQN_Agent(player=1, env=env,parametes_path=None)
    player2_hat = DQN_Agent(player=1, env=env, train=False)
    Q2 = player2.DQN
    Q2_hat = Q2.copy()
    player2_hat.DQN = Q2_hat
    
    buffer1 = ReplayBuffer(path=None) 
    buffer2 = ReplayBuffer(path=None) 
    
    tester1 = Tester(player1=player1, player2=Random_Agent(), env=env)
    tester2 = Tester(player1=Random_Agent(), player2= player2, env=env)
    
    results = [] #results_file['results'] # []
    avgLosses1, avgLosses2 = [], [] #results_file['avglosses']     #[]
    avgLoss1, avgLoss2 = 0, 0 #avgLosses[-1] #0
    loss = 0
    res = 0
    best_res = -200
    loss_count1, loss_count2 = 0, 0
    
    # random_results = [] #torch.load(random_results_path)   # []
    best_random1, best_random2 = 0, 0
    best_random_params1, best_random_params2 = None, None

    # init optimizer
    optim1 = torch.optim.Adam(Q1.parameters(), lr=learning_rate)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optim1,10000*30, gamma=0.50)
    optim2 = torch.optim.Adam(Q2.parameters(), lr=learning_rate)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optim2,10000*30, gamma=0.50)
    
    file_num = 1111
    #region ######## checkpoint Load ############
    checkpoint_path = f"Data/checkpoint{file_num}.pth"
    buffer_path = f"Data/buffer{file_num}.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']+1
        player1.DQN.load_state_dict(checkpoint['model_state_dict1'])
        player2.DQN.load_state_dict(checkpoint['model_state_dict2'])
        player1_hat.DQN.load_state_dict(checkpoint['model_state_dict1'])
        optim1.load_state_dict(checkpoint['optimizer_state_dict1'])
        scheduler1.load_state_dict(checkpoint['scheduler_state_dict1'])
        player2_hat.DQN.load_state_dict(checkpoint['model_state_dict2'])
        optim2.load_state_dict(checkpoint['optimizer_state_dict2'])
        scheduler2.load_state_dict(checkpoint['scheduler_state_dict2'])
        buffer1 = torch.load(buffer_path)
        buffer2 = torch.load(buffer_path)
        results = checkpoint['results']
        avgLosses1 = checkpoint['avgLosses1']
        avgLosses2 = checkpoint['avgLosses2']
        best_res = checkpoint['best_res']
        best_random_params1 = checkpoint['best_random_params1']
        best_random_params2 = checkpoint['best_random_params2']
        best_random1 = checkpoint['best_random1']
        best_random2 = checkpoint['best_random2']
    player1.DQN.train()
    player1_hat.DQN.eval()
    player2.DQN.train()
    player2_hat.DQN.eval()
    

    #endregion
    
    #region ################ Wandb.init #####################
    wandb.init(
        # set the wandb project where this run will be logged
        project="Mancala",
        resume=False, 
        id=f'Mancala{file_num}',
        # track hyperparameters and run metadata
        config={
        "name": f"Mancala{file_num}",
        "checkpoint": checkpoint_path,
        "learning_rate": learning_rate,
        "Model": str(player1.DQN),
        "Schedule1": f'{str(scheduler1.step_size)} gamma={str(scheduler1.gamma)}',
        "Schedule2": f'{str(scheduler2.step_size)} gamma={str(scheduler2.gamma)}',
        "epochs": epochs,
        "start_epoch": start_epoch,
        "decay": epsiln_decay,
        "gamma": 0.99,
        "batch_size": batch_size, 
        "C": C,
        'player1 ': 'dqn1',
        'player2 ': 'dqn2'
        }
    )
    #endregion
    
    for epoch in range(start_epoch, epochs):
        print(f'epoch = {epoch}', end='\r')
        state_1.init_Board()
        step = 0
        while not env.is_end_state(state_1):
            #region ######### Sample Environement #########
            step += 1
            action_1 = player1.get_action(state=state_1, epoch=epoch)
            is_another_turn = state_1.is_another_turn
            after_state_1 = env.get_next_state(state=state_1, action=action_1, is_DQN = True)
            reward_1, end_of_game_1 = env.reward(state=after_state_1, is_another_turn = is_another_turn)
            if step > 1:
                buffer2.push(state_2, action_2, -reward_1-reward_2, after_state_1, end_of_game_1)
            if end_of_game_1:
                if reward_1 > 0:
                    res += 1
                elif reward_1 < 0:
                    res -= 1
                buffer1.push(state_1, action_1, reward_1, after_state_1, True)
                break

            state_2 = after_state_1
            action_2 = player2.get_action(state=state_2, epoch=epoch)
            is_another_turn = state_2.is_another_turn
            after_state_2 = env.get_next_state(state=state_2, action=action_2, is_DQN = True)
            reward_2, end_of_game_2 = env.reward(state=after_state_2, is_another_turn = is_another_turn)
            if end_of_game_2:
                if reward_2 > 0:
                    res += 1
                elif reward_2 < 0:
                    res -= 1
                buffer2.push(state_2, action_2, -reward_2, after_state_2, True)
            buffer1.push(state_1, action_1, reward_2+reward_1, after_state_2, end_of_game_2)
            state_1 = after_state_2

            if len(buffer1) < MIN_Buffer:
                continue
            #endregion

            #region ######### Train NN for player1 #########
            states, actions, rewards, next_states, dones = buffer1.sample(batch_size)
            Q_values = Q1(states, actions)
            next_actions = player1.get_Actions(next_states, dones) # DDQN ; DQN - player_hat.get_Actions(next_states, dones) 
            with torch.no_grad():
                Q_hat_Values = Q1_hat(next_states, next_actions) 

            loss = Q1.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim1.step()
            optim1.zero_grad()
            scheduler1.step()
            
            if loss_count1 <= 1000:
                avgLoss1 = (avgLoss1 * loss_count1 + loss.item()) / (loss_count1 + 1)
                loss_count1 += 1
            else:
                avgLoss1 += (loss.item()-avgLoss1)* 0.0001 
            #endregion

            #region ######## Train NN for player2 ########
            states, actions, rewards, next_states, dones = buffer2.sample(batch_size)
            Q_values = Q2(states, actions)
            next_actions = player2.get_Actions(next_states, dones) # DDQN ; DQN - player_hat.get_Actions(next_states, dones) 
            with torch.no_grad():
                Q_hat_Values = Q2_hat(next_states, next_actions) 

            loss = Q2.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim2.step()
            optim2.zero_grad()
            scheduler2.step()

            if loss_count2 <= 1000:
                avgLoss2 = (avgLoss2 * loss_count2 + loss.item()) / (loss_count2 + 1)
                loss_count2 += 1
            else:
                avgLoss2 += (loss.item()-avgLoss2)* 0.00001 
            #endregion

        if epoch % C == 0:
            Q1_hat.load_state_dict(Q1.state_dict())
            Q2_hat.load_state_dict(Q2.state_dict())

        if (epoch+1) % 100 == 0:
            avgLosses1.append(avgLoss1)
            avgLosses2.append(avgLoss2)
            results.append(res)
            best_res = max(best_res, res)
            wandb.log ({
                "result": res,
                "avgLoss1": avgLoss1,
                "avgLoss2": avgLoss2,
            })
            res = 0
        
        #region ### Finding best model ###########
        if (epoch+1) % 100 == 0:
            test1 = tester1(100)
            test_score1 = test1[0]-test1[1]
            if best_random1 < test_score1:
                best_random1 = test_score1
                best_random_params1 = player1.DQN.state_dict().copy()
            print("dqn1 vs rnd: ", test1)
            test2 = tester2(100)
            test_score2 = test2[1]-test2[0]
            if best_random2 < test_score2:
                best_random2 = test_score2
                best_random_params2 = player2.DQN.state_dict().copy()
            print("dqn2 vs rnd: ", test2)
            wandb.log ({
              "random1":test_score1,
              "random2":test_score2,              
            })
        #endregion 

        #region ### Saving checkpoint ###########
        if epoch % 500 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict1': player1.DQN.state_dict(),
                'model_state_dict2': player2.DQN.state_dict(),
                'optimizer_state_dict1': optim1.state_dict(),
                'scheduler_state_dict1': scheduler1.state_dict(),
                'optimizer_state_dict2': optim2.state_dict(),
                'scheduler_state_dict2': scheduler2.state_dict(),
                'results': results,
                'avgLosses1':avgLosses1,
                'avgLosses2':avgLosses2,
                'best_res': best_res,
                'best_random_params1': best_random_params1,
                'best_random_params2': best_random_params2,
                'best_random1': best_random1,
                'best_random2': best_random2
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(buffer1, buffer_path)
            torch.save(buffer2, buffer_path)
        #endregion

        print (f'epoch = {epoch} step = {step} loss = {loss:.5f} avgloss1 = {avgLoss1:.5f} avgloss2 = {avgLoss2:.5f}', end=" ")
        print (f'learning rate1 = {scheduler1.get_last_lr()[0]} path = {checkpoint_path} res = {res}', end=" " )
        print (f'learning rate2 = {scheduler2.get_last_lr()[0]}')

    torch.save(checkpoint, checkpoint_path)
    torch.save(buffer1, buffer_path)
    torch.save(buffer2, buffer_path)
    

if __name__ == '__main__':
    main()