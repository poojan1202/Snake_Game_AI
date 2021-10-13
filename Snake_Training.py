import random as rd
import gym
import numpy as np
import pickle as pk
from numpy import save

import gym_snake

# Construct Environment
env = gym.make('snake-v0')


def action_space(action, dir):
    # action{0:Left, 1:forward, 2:right}
    # Env_actions and directions {0:Up, 1:Right, 2:Down, 3:Left}
    # Head Upwards
    if dir == 0:
        if action==0:
            env_action = 3

        elif action==1:
            env_action = 0

        elif action==2:
            env_action = 1

    # Head Rightwards
    if dir==1:
        if action==0:
            env_action = 0

        elif action==1:
            env_action = 1

        elif action == 2:
            env_action = 2

    # Head Downwards
    if dir==2:
        if action==0:
            env_action = 1

        elif action==1:
            env_action = 2

        elif action == 2:
            env_action = 3

    # Head Leftwards
    if dir==3:
        if action==0:
            env_action = 2

        elif action==1:
            env_action = 3

        elif action == 2:
            env_action = 0

    return env_action


def state_space(head_loc,rew_loc, body_loc, dir):
    """

    :param head_loc:
    :param rew_loc:
    :param dir:
    (0) : obstacle above head (1 else 0), (1) : obstacle below head (1 else 0),
    (2) : obstacle right to head (1 else 0), (3) : obstacle left to head (1 else 0)
    (4) : reward in front of head = 1, behind head = -1, else 0
    (5) : reward right to head = 1, below left to head = -1, else 0
    :return:
    """
    state = [0, 0, 0, 0, 0]
    head_x, head_y = head_loc[0],head_loc[1]
    rew_x, rew_y = rew_loc[0],rew_loc[1]
    x_dis = rew_x - head_x
    y_dis = head_y - rew_y
    if dir==0:

        if x_dis>0:
            state[4] = 1
        elif x_dis<0:
            state[4] = -1
        else:
            state[4] = 0
        if y_dis>0:
            state[3] = 1
        elif y_dis<0:
            state[3] = -1
        else:
            state[3] = 0
        if abs(x_dis) == abs(y_dis):
            if x_dis > 0 and y_dis > 0:
                state[3] = 1
                state[4] = 1
            elif x_dis > 0 and y_dis < 0:
                state[4] = 1
                state[3] = -1
            elif x_dis < 0 and y_dis < 0:
                state[3] = -1
                state[4] = -1
            elif x_dis < 0 and y_dis > 0:
                state[4] = -1
                state[3] = 1

    if dir==1:

        if x_dis>0:
            state[3] = 1
        elif x_dis<0:
            state[3] = -1
        else:
            state[3] = 0
        if y_dis>0:
            state[4] = -1
        elif y_dis<0:
            state[4] = 1
        else:
            state[4] = 0
        if abs(x_dis) == abs(y_dis):
            if x_dis > 0 and y_dis > 0:
                state[4] = -1
                state[3] = 1
            elif x_dis > 0 and y_dis < 0:
                state[3] = 1
                state[4] = 1
            elif x_dis < 0 and y_dis < 0:
                state[4] = 1
                state[3] = -1
            elif x_dis < 0 and y_dis > 0:
                state[3] = -1
                state[4] = -1

    if dir==2:

        if x_dis>0:
            state[4] = -1
        elif x_dis<0:
            state[4] = 1
        else:
            state[4] = 0
        if y_dis>0:
            state[3] = -1
        elif y_dis<0:
            state[3] = 1
        else:
            state[3] = 0
        if abs(x_dis) == abs(y_dis):
            if x_dis > 0 and y_dis > 0:
                state[3] = -1
                state[4] = -1
            elif x_dis > 0 and y_dis < 0:
                state[4] = -1
                state[3] = 1
            elif x_dis < 0 and y_dis < 0:
                state[4] = 1
                state[3] = 1
            elif x_dis < 0 and y_dis > 0:
                state[3] = -1
                state[4] = 1

    if dir==3:

        if x_dis>0:
            state[3] = -1
        elif x_dis<0:
            state[3] = 1
        else:
            state[3] = 0
        if y_dis>0:
            state[4] = 1
        elif y_dis<0:
            state[4] = -1
        else:
            state[4] = 0
        if abs(x_dis) == abs(y_dis):
            if x_dis > 0 and y_dis > 0:
                state[4] = 1
                state[3] = -1
            elif x_dis > 0 and y_dis < 0:
                state[3] = -1
                state[4] = -1
            elif x_dis < 0 and y_dis < 0:
                state[4] = -1
                state[3] = 1
            elif x_dis < 0 and y_dis > 0:
                state[3] = 1
                state[4] = 1


    for i in range(0, len(body_loc)):
        x,y = body_loc[i][0],body_loc[i][1]
        if dir==0:
            if head_x - x == 1 and head_y - y == 0:
                state[0] = 1
            if head_x - x == -1 and head_y - y == 0:
                state[2] = 1
            if head_x - x == 0 and head_y - y == 1:
                state[1] = 1

        if dir==1:
            if head_x - x == -1 and head_y - y == 0:
                state[1] = 1
            if head_x - x == 0 and head_y - y == 1:
                state[0] = 1
            if head_x - x == 0 and head_y - y == -1:
                state[2] = 1

        if dir==2:
            if head_x - x == 1 and head_y - y == 0:
                state[2] = 1
            if head_x - x == -1 and head_y - y == 0:
                state[0] = 1
            if head_x - x == 0 and head_y - y == -1:
                state[1] = 1

        if dir==3:
            if head_x - x == 1 and head_y - y == 0:
                state[1] = 1
            if head_x - x == 0 and head_y - y == 1:
                state[2] = 1
            if head_x - x == 0 and head_y - y == -1:
                state[0] = 1

    if dir==0:
        if head_x == 0:
            state[0] = 1
        if head_x == 14:
            state[2] = 1
        if head_y == 0:
            state[1] = 1
    if dir==1:
        if head_x == 14:
            state[1] = 1
        if head_y == 0:
            state[0] = 1
        if head_y == 14:
            state[2] = 1

    if dir==2:
        if head_x == 0:
            state[2] = 1
        if head_x == 14:
            state[0] = 1
        if head_y == 14:
            state[1] = 1

    if dir==3:
        if head_x == 0:
            state[1] = 1
        if head_y == 0:
            state[2] = 1
        if head_y == 14:
            state[0] = 1


    return state


def epsilon_greedy(idx1,epsilon):
    p = rd.random()
    if p<epsilon:
        a = rd.randint(0,2)
    else:
        a = greedy_action[0][idx1]
    return int(a)


def max_act_value(act_vals):
    i = -1
    maximum = act_vals[0]
    max_idx = 0
    for values in act_vals:
        i += 1
        if values >= maximum:
            maximum = values
            max_idx = i
    return max_idx

# List containing all states that are encountered throughout all episodes

st_list = []

action_values = np.zeros((1,3))
greedy_action = np.zeros((1,1))

epsilon = 1
for ep in range(0,3000):
    if ep>=500:
        epsilon-= 1/2500
    print('Episode : ',ep)
    done = False
    observation = env.reset()  # Constructs an instance of the game
    for i in range(0, 150):
        for j in range(0, 150):
            if observation[i][j][2] == 255:
                rew_location = [j//10,i//10]
                break

    # Controller
    game_controller = env.controller

    # Grid
    grid_object = game_controller.grid
    grid_pixels = grid_object.grid

    # Snake(s)
    snakes_array = game_controller.snakes
    snake_object1 = snakes_array[0]

    snake_dir_1 = snake_object1.direction
    head_location_1 = snake_object1.head
    # print(head_location)

    st = state_space(head_location_1, rew_location, snake_object1.body, snake_dir_1)

    if st in st_list:
        idx1 = st_list.index(st)
    else:
        if ep==0:
            st_list.append(st)
            idx1 = st_list.index(st)
        else:
            st_list.append(st)
            idx1 = st_list.index(st)
            action_values.resize((len(st_list),3))
            greedy_action.resize((1,len(st_list)))



    while not done:

        # env.render()
        action = epsilon_greedy(idx1,epsilon)
        take_action = action_space(action, snake_dir_1)

        obs,rew,done,info = env.step(take_action)

        snake_dir_2 = snake_object1.direction
        head_location_2 = snake_object1.head


        if rew>0:
            for i in range(0, 150):
                for j in range(0, 150):
                    if obs[i][j][2] == 255:
                        rew_location = [j // 10, i // 10]
                        break

        st2 = state_space(head_location_2, rew_location, snake_object1.body, snake_dir_2)

        if st2 in st_list:
            idx2 = st_list.index(st2)
        else:
            st_list.append(st2)
            idx2 = st_list.index(st2)
            action_values.resize((len(st_list),3))
            greedy_action.resize((1,len(st_list)))
        amax = max_act_value(action_values[idx2])

        action_values[idx1][action] += 0.2*(rew + 0.9*action_values[idx2][amax] - action_values[idx1][action])

        st = st2
        idx1 = idx2
        snake_dir_1 = snake_dir_2

        if rew<0:
            done = True

        if ep == 2999:
            env.render()

    a,b = action_values.shape
    for i in range(0,a):
        maxi = action_values[i][0]
        g_act = 0
        for j in range(0,b):
            if action_values[i][j]>= maxi:
                maxi = action_values[i][j]
                g_act = j
        greedy_action[0][i] = g_act

file_name = 'states_2.pkl'
open_file = open(file_name,'wb')
pk.dump(st_list,open_file)
open_file.close()

save('optimal_action_2.npy',greedy_action)


