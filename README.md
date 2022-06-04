# RL Stuff

My reinforcement learning experiments

## Dodgy

To start getting familiar with OpenAI Gym and reinforcement learning, I wrote
a very simple game called Dodgy. The point of the game is to move left and
right to avoid objects falling down from above. In order to facilitate quick
experimentation with different learning algorithms, the game has a very low
number of possible states and actions. 

Run [dodgy/dodgy_env.py](dodgy/dodgy_env.py) directly with
`$ python dodgy_env.py` to play the game yourself. Left and right arrow keys
control the blue square, and you need to avoid touching the red squares.

To train a policy gradient network to play the game, you can run
[dodgy/dodgy_policy_gradient.py](dodgy/dodgy_policy_gradient.py). There are
several optional features like rendering the environment and saving the trained
network to a faile, which are enabled with command line arguments. Run
`$ python dodgy_policy_gradient.py --help` for more information.

Training the policy gradient network to play Dodgy usually only takes a few
minutes for it to become fairly proficient, and only a few hours to become
practically flawless. However, it's fairly unstable--as it continues to train,
it may go through periods where its proficiency drops all the way back down to
where it started before training.

