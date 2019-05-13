import gym
import time

env = gym.make('CartPole-v1')
action = 0


def key_press(key, mod):
    global action
    a = int(key - ord('0'))-1
    action = a


def key_release(key, mod):
    None


def main():
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    while 1:
        env.reset()
        total_reward = 0
        total_timesteps = 0
        global action
        action = env.action_space.sample()
        print(action)
        while 1:
            a = action
            observation, reward, done, info = env.step(a)
            total_reward += reward
            env.render()
            if done:
                break
            time.sleep(0.08)
        print("Total time steps %i and reward %0.2f" % (total_timesteps, total_reward))


if __name__ == "__main__":
    main()
