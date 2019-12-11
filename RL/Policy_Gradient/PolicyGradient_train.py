import gym
from PolicyGradient import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400
RENDER = False

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

print("action space:", env.action_space)
print("observation space:", env.observation_space, " , high:", env.observation_space.high, " , low:", env.observation_space.low)

pg = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
)

for i_episode in range(3000):
    observation = env.reset()
    while True:
        if RENDER:
            env.render()
        action = pg.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        pg.store_transition(observation, action, reward)

        if done:
            ep_rs_sum = sum(pg.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True
            print("episode {:5d}".format(i_episode), " reward:", int(running_reward))
            
            vt = pg.learn()

            if i_episode == 0:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break
        
        observation = observation_
