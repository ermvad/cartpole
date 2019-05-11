import gym
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

RIGHT_CMD = [0, 1]
LEFT_CMD = [1, 0]

MIN_REWARD = 100

env = gym.make('CartPole-v1')


def key_press(key, mod):
    push = env.action_space.sample()
    env.step(push)


def key_release(key, mod):
    None


def play_random_games(games=100):
    all_movements = []

    for episode in range(games):
        episode_reward = 0
        current_game_data = []
        env.reset()

        action = env.action_space.sample()

        while True:
            observation, reward, done, info = env.step(action)
            action = env.action_space.sample()
            current_game_data.append(np.hstack((observation, LEFT_CMD if action == 0 else RIGHT_CMD)))

            if done:
                break

            episode_reward += reward

        if episode_reward >= MIN_REWARD:
            print('.', end='')
            all_movements.extend(current_game_data)

    dataframe = pd.DataFrame(
        all_movements,
        columns=['cart_position', 'cart_velocity', 'pole_angle', 'pole_velocity_at_tip', 'action_to_left', 'action_to_right']
    )

    dataframe['action_to_left'] = dataframe['action_to_left'].astype(int)
    dataframe['action_to_right'] = dataframe['action_to_right'].astype(int)

    return dataframe


def generate_ml(dataframe):
    model = Sequential()
    model.add(Dense(64, input_dim=4, activation='relu'))
    # model.add(Dense(128,  activation='relu'))
    # model.add(Dense(128,  activation='relu'))
    model.add(Dense(64,  activation='relu'))
    model.add(Dense(32,  activation='relu'))
    model.add(Dense(2,  activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    model.fit(
        dataframe[['cart_position', 'cart_velocity', 'pole_angle', 'pole_velocity_at_tip']],
        dataframe[['action_to_left', 'action_to_right']],
        epochs=30
    )

    return model


def play_game(ml_model, games=100):
    for i_episode in range(games):
        episode_reward = 0
        observation = env.reset()

        while True:
            env.render()

            current_action_pred = ml_model.predict(observation.reshape(1, 4))

            current_action = np.argmax(current_action_pred)

            observation, reward, done, info = env.step(current_action)

            if done:
                episode_reward += 1
                print(f"Episode finished after {i_episode+1} steps", end='')
                break

            episode_reward += 1

        print(f" Score = {episode_reward}")


def main():
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    print("Playingr random games...")
    df = play_random_games(games=20000)

    print("Training NN model...")
    ml_model = generate_ml(df)

    print("Playing games with NN...")
    play_game(ml_model=ml_model, games=100)


if __name__ == "__main__":
    main()
