import numpy as np
import torch
import matplotlib.pyplot as plt

Files_num = [1111]

results_path = []
for num in Files_num:
    file = f'Data/checkpoint{num}.pth'
    results_path.append(file)

checkpoints = []
for path in results_path:
    checkpoints.append(torch.load(path))


# avg = {}
# game_num = 25
# for index, cp in enumerate(checkpoints):
#     avg_score = 0
#     game = 1
#     avg[index] = []
#     for score in cp["avg_score"]:
#         avg_score = (avg_score * (game-1) + score) / game
#         if game == game_num:
#             avg[index].append(avg_score)
#             game = 0
#         game += 1
        
        

for i in range(len(checkpoints)):
    fig, ax_list = plt.subplots(3,1, figsize = (10,6))
    fig.suptitle(f'{results_path[i]} epochs: {checkpoints[i]["epoch"]}')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    ax_list[0].plot(checkpoints[i]['results'])
    ax_list[0].title.set_text("results")
    ax_list[1].plot(checkpoints[i]['avgLosses1'])
    ax_list[1].title.set_text("avgLosses1")
    ax_list[2].plot(checkpoints[i]['avgLosses2'])
    ax_list[2].title.set_text("avgLosses2")
    # ax_list[2].plot(avg[i])
    # ax_list[2].title.set_text("average score")
    # ax_list[3].plot(checkpoints[i]['avg_score'])
    # ax_list[3].title.set_text("average score")
    plt.tight_layout()

plt.show()