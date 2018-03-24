import random
from collections import deque

import gym
import keras
import numpy as np
from keras import Sequential
from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import RMSprop

# Change to true to load a saved model
LOAD = False

EPISODES = 5
# SGD updates are sampled from this number of most recent frames
MEMORY = 10000000
# 5 minutes running on 60 FPS:
MAX_TIME = 18000
# Deterministic-v4 is the version used by the original deepmind paper, which helps to deal with atari's limitations
GAME = 'BreakoutDeterministic-v4'

memory = deque(maxlen=MEMORY)
env = gym.make(GAME)

state_size = env.observation_space.shape
action_size = env.action_space.n

done = False


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Discount factor
        self.gamma = 0.99

        # Exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.learning_rate = 0.001

        # RMSProp hyperparameters
        self.optimizer_lr = 0.00025
        self.optimizer_rho = 0.95
        self.optimizer_epsilon = 0.01

        # Image Pre-processing
        self.frames_per_action = 30
        self.frames_per_state = 4

        self.batch_size = 32
        # Target network update Frequency
        self.C = 10000
        self.C_counter = 0

        if LOAD:
            self.model = keras.models.load_model('hlctdrl-model', custom_objects={'huber_loss': self.huber_loss})
            self.model_target = keras.models.load_model('hlctdrl-model-target',
                                                        custom_objects={'huber_loss': self.huber_loss})
        else:
            self.model = self.build_model(state_size, action_size)
            self.model_target = self.model

        self.Q = np.ones(action_size)
        self.Q_hat = self.Q

    def build_model(self, state_size, action_size):
        model = Sequential()
        # Convolutional layers as per deepmind paper
        model.add(Conv2D(16, 8, input_shape=state_size, strides=4, activation='relu'))
        model.add(Conv2D(32, 4, strides=2, activation='relu'))
        # Flatten the convolutional layers to pass trough final layer
        model.add(Flatten())
        # Linear output layer
        model.add(Dense(action_size))
        model.compile(optimizer=RMSprop(lr=self.optimizer_lr, rho=self.optimizer_rho, epsilon=self.optimizer_epsilon),
                      loss=self.huber_loss)
        return model

    def act(self, env, frame, timestep, memory):
        # Explore less randomly over time
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay = self.epsilon

        # Take a random exploratory action sometimes, otherwise take best action it can
        if random.random() < self.epsilon:
            action = np.random.randint(0, env.action_space.n, size=1)[0]
        else:
            # Only predict once every few frames to save computational
            # time and simulate a better human learning experience
            self.Q = self.model.predict(frame)
            action = np.argmax(self.Q)

        frame_new, reward, done, _ = env.step(action)
        frame_new = self.preprocess(frame_new)

        memory.append((frame, action, reward, frame_new, done))

        # Train on the past transitions
        if timestep / self.frames_per_state >= self.batch_size:
            self.fit_batch(memory)

        # Every few steps reset target model to trained model
        self.C_counter += 1
        if self.C_counter >= self.C:
            self.C_counter = 0
            self.model_target = self.copy_model(self.model)

    def fit_batch(self, memory):
        minibatch = random.sample(memory, self.batch_size)
        for i in range(0, self.batch_size):
            frame = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            frame_new = minibatch[i][3]
            done = minibatch[i][4]

            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.model_target.predict(frame_new))

            target_future = self.model_target.predict(frame)
            target_future[0][action] = target

            self.model.fit(frame, target_future, epochs=1, verbose=0)

    def copy_model(self, model):
        model.save('tmp_model')
        return keras.models.load_model('tmp_model', custom_objects={'huber_loss': self.huber_loss})

    def preprocess(self, img):
        # Take the max between two frames to eliminate problems with atari flickering
        temp, _, _, _ = env.step(np.argmax(agent.Q))
        img = np.fmax(img, temp)
        # RGB to Grayscale
        # img = Image.fromarray(img).convert('L')
        # plt.figure()
        # plt.imshow(img)
        # plt.show()

        img = np.expand_dims(img, axis=0)

        return img

    def huber_loss(self, a, b, in_keras=True):
        error = a - b
        quadratic_term = error * error / 2
        linear_term = abs(error) - 1 / 2
        use_linear_term = (abs(error) > 1.0)
        if in_keras:
            use_linear_term = K.cast(use_linear_term, 'float32')
        return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term


agent = Agent(state_size, action_size)

for episode in range(EPISODES):
    print('Episode: {}  Reward: {}'.format(episode, 1))
    frame = env.reset()
    agent.model.save('hlctdrl-model')
    agent.model_target.save('hlctdrl-model-target')

    for timestep in range(MAX_TIME):
        if episode == (EPISODES - 1):
            env.render()

        # Repeat same action a few times before moving on to simulate human experience
        # and be more computationally efficient
        if timestep % agent.frames_per_state == 0 and timestep >= agent.frames_per_state:
            frame = agent.preprocess(frame)
            agent.act(env, frame, timestep, memory)
        else:
            frame, _, _, _ = env.step(np.argmax(agent.Q))

        if done:
            env.render(close=True)
            break
