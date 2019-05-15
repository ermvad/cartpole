import gym
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

MOVEMENT_RIGHT = [0, 1]
MOVEMENT_LEFT = [1, 0]

MIN_REWARD = 60

env = gym.make('CartPole-v1')


def key_press(key, mod):
    push = env.action_space.sample()
    env.step(push)


def key_release(key, mod):
    None


def play_random_games(games):
    x_train = np.zeros((0, 4))
    y_train = np.zeros((0, 2))

    for episode in range(games):
        episode_reward = 0
        current_observations = np.zeros((0, 4))
        current_actions = np.zeros((0, 2))
        env.reset()

        action = env.action_space.sample()

        while True:
            observation, reward, done, info = env.step(action)
            action = env.action_space.sample()
            current_observations = np.append(current_observations, [observation], axis=0)
            current_actions = np.append(current_actions, [(MOVEMENT_LEFT if action == 0 else MOVEMENT_RIGHT)], axis=0)

            if done:
                break

            episode_reward += reward

        if episode_reward >= MIN_REWARD:
            x_train = np.vstack((x_train, current_observations))
            y_train = np.vstack((y_train, current_actions))

    return x_train, y_train


def generate_model(x_train, y_train):
    model = Sequential()
    model.add(Dense(64, input_dim=4, activation='relu'))
    # model.add(Dense(128,  activation='relu'))
    # model.add(Dense(128,  activation='relu'))
    model.add(Dense(64,  activation='relu'))
    model.add(Dense(32,  activation='relu'))
    model.add(Dense(2,  activation='sigmoid'))

    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=10)

    return model


def play_game(model, games):
    for episode in range(games):
        episode_reward = 0
        observation = env.reset()

        while True:
            env.render()

            current_action_pred = model.predict(observation.reshape(1, 4))

            current_action = np.argmax(current_action_pred)

            observation, reward, done, info = env.step(current_action)

            if done:
                episode_reward += 1
                print("Episode", episode, "finished after", episode_reward, "steps")
                break

            episode_reward += 1

        print("Score =", episode_reward)


def main():
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    print("Playing random games")
    x_train, y_train = play_random_games(10000)

    print("Training NN model")
    model = generate_model(x_train, y_train)

    print("Playing games with NN")
    play_game(model, 100)


if __name__ == "__main__":
    main()
