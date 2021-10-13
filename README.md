# Snake_Game_AI
This project is focussed at solving the Snake Game using Tabular Reinforcement Learning Techniques

## Approach
### State Space
The State of the agent is represented as a list of size 5, where the first 3 indices represent position of obstacle in immediate neighbourhood (left, top, right) grid of the snake head, and the last 2 indices represent the direction of fruit relative to the Snake's Head. 
`state = [0,0,0,0,0]`
- Obstacles
    - The four walls and the Snake's body (other than the head) is treated as an obstacle.
    - The grids adjacent to the Snake's Head (Immediate Left, Top and Right grids) are considered, where the Snake could move in the next step.
            
            state[0] = 1    if obstacle is present at left grid
            state[1] = 1    if obstacle is present at top grid
            state[2] = 1    if obstacle is present at right grid
            
- Relative Position of Fruit
    - The grid is divided into 4 quadrants, where the Snake's Head is taken as origin and its direction as the positive Y-axis. Thus, with respect to the Head, fruit lies in any of these quadrants (or on the axis).
            
            state[3] =   1    if fruit is located above Head
                        -1    if fruit is located below Head
                         0    if fruit located on X-Axis`
            state[4] =   1    if fruit is Rightwards wrt Head
                        -1    if fruit is Leftwards wrt Head
                         0    if fruit is on Y-Axis wrt Head
This parameters forms a state space of 2x2x2x3x3 = 72 states

### Action Space
In the given Environment, Actions are predefined with respect to the grid, which are to:
- Move UP
- Move RIGHT
- Move DOWN
- Move LEFT
which do not depend on the agent's direction. To reduce the number of state-action pairs, the action space is redefined, knowing that the agent could not move backward under any case.
Thus, knowing the direction in which the Snake's Head is present, 3 action are defined as:
- Move Left
- Move Forward
- Move Right
Thus, these actions together with the state space gives 72x3 = 216 state-action pairs, each state with 3 possible actions.
### Rewards
A +1 reward is returned when a snake eats a fruit.

A -1 reward is returned when a snake dies when hits an obstacle.

No extra reward is given for victory snakes in plural play.
### Estimating Action-Values
While numerous Tabular RL methods such as Sarsa(λ), Backward Sarsa and Q-learning can be applied to etimate action-values Q(s,a), here the agent is trained using Q-Learning, which is an off-Policy Control Method, where the agent evaluates target policy π(s/a), while following a behaviour policy μ(s/a).
- Behaviour Policy μ(s/a) is ε-greedy wrt Q(s,a)
- Target Policy π(s/a) is greedy wrt Q(s,a)
# Environment
[Gym-Snake](https://github.com/grantsrb/Gym-Snake)
#### Created in response to OpenAI's [Requests for Research 2.0](https://blog.openai.com/requests-for-research-2/)

## Description
gym-snake is a multi-agent implementation of the classic game [snake](https://www.youtube.com/watch?v=wDbTP0B94AM) that is made as an OpenAI gym environment.

The two environments this repo offers are snake-v0 and snake-plural-v0. snake-v0 is the classic snake game. See the section on SnakeEnv for more details. snake-plural-v0 is a version of snake with multiple snakes and multiple snake foods on the map. See the section on SnakeExtraHardEnv for more details. 

Many of the aspects of the game can be changed for both environments. See the Game Details section for specifics on [Gym-Snake](https://github.com/grantsrb/Gym-Snake).

## Dependencies
- pip
- gym
- numpy
- matplotlib

## Installation
1. Clone this repository
2. Navigate to the cloned repository
3. Run command `$ pip install -e ./`


