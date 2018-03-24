import random
from collections import deque

import gym
import keras
import numpy as np
from keras import Model
from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten, Lambda, Input, multiply
from keras.optimizers import RMSprop

# Change to true to load a saved model
LOAD = True

EPISODES = 1000
# SGD updates are sampled from this number of most recent frames
MEMORY = 10000000
# 5 minutes running on 60 FPS, considering we only play every 4 frames:
MAX_TIME = 4500
# Deterministic-v4 is the version used by the original deepmind paper, which helps to deal with atari's limitations
GAME = 'BreakoutDeterministic-v4'

memory = deque(maxlen=MEMORY)
env = gym.make(GAME)

state_size = env.observation_space.shape
# Halve size to accommodate for image preprocessing
state_size = np.array(state_size)
state_size[0] = state_size[0] / 2
state_size[1] = state_size[1] / 2

action_size = env.action_space.n


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
        self.frames_per_state = 4

        self.random_init_counter = 0
        self.random_init = 50000
        self.batch_size = 32
        # Target network update Frequency
        self.C = 10000 / self.frames_per_state
        self.C_counter = 0

        # Just for debugging
        self.debug_reward = 0

        if LOAD:
            self.model = keras.models.load_model('hlctdrl-model', custom_objects={'huber_loss': self.huber_loss})
        else:
            self.model = self.build_model(state_size, action_size)

        self.model_target = self.copy_model(self.model)

    def build_model(self, state_size, action_size):
        frames_input = Input(state_size, name='frames')
        actions_input = Input((1,), name='mask')

        # Normalization layer for more efficiency
        normalized = Lambda(lambda x: x / 255.0)(frames_input)

        # Convolutional layers as per deepmind paper
        conv1 = Conv2D(16, 8, strides=4, activation='relu')(normalized)
        conv2 = Conv2D(32, 4, strides=2, activation='relu')(conv1)
        # Flatten the convolutional layers to pass trough final layer
        flattened = Flatten()(conv2)
        hidden = Dense(256, activation='relu')(flattened)
        # Linear output layer
        output = Dense(action_size)(hidden)
        filtered_output = multiply([output, actions_input])

        model = Model(input=[frames_input, actions_input], output=filtered_output)
        optimizer = RMSprop(lr=self.optimizer_lr, rho=self.optimizer_rho, epsilon=self.optimizer_epsilon)

        model.compile(optimizer, self.huber_loss)

        return model

    def act(self, env, frame):
        # Explore less randomly over time
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay = self.epsilon

        # Take a random exploratory action sometimes, otherwise take best action it can
        if random.random() < self.epsilon:
            action = np.random.randint(0, env.action_space.n, size=1)[0]
        else:
            # Only predict once every few frames to save computational
            # time and simulate a better human learning experience
            action = np.argmax(self.model.predict(frame))

        return action

    def fit_batch(self, memory):
        minibatch = random.sample(memory, self.batch_size)
        for i in range(self.batch_size):
            frame = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            frame_new = minibatch[i][3]
            done = minibatch[i][4]

            action = np.expand_dims(action, axis=0)
            action[0] = i
            if done:
                target = reward
            else:
                action_next = self.model_target.predict([frame_new, np.ones(1, )])
                target = reward + self.gamma * np.argmax([action_next])

            loss = np.square(target - self.model_target.predict([frame, action]))
            agent.debug_reward = target
            self.model.fit([frame, action], loss, batch_size=1, verbose=0)

    # Consistent way to copy models with any keras version
    def copy_model(self, model):
        model.save('tmp_model')
        return keras.models.load_model('tmp_model', custom_objects={'huber_loss': self.huber_loss})

    def preprocess(self, img, img2):
        # Take the max between two frames to eliminate problems with atari flickering
        img = np.fmax(img, img2)
        # Downscale image
        img = img[::2, ::2]
        # RGB to Grayscale
        img[..., :] = np.mean(img, axis=2, keepdims=True).astype(np.uint8)

        # Expand dims to fit the Conv2D input parameters
        img = np.expand_dims(img, axis=0)

        # Uncomment to see individual preprocessed frames
        # plt.figure()
        # plt.imshow(img)
        # plt.show()

        # Don't normalize yet to save RAM

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
    print('Episode: {}  Reward: {}'.format(episode, agent.debug_reward))
    frame = env.reset()
    frame2 = frame

    for timestep in range(MAX_TIME):
        # Render the last iteration and every time a model is saved/the target model gets updated
        if episode == (EPISODES - 1) or agent.C_counter >= agent.C:
            env.render()

        # Repeat same action a few frames before next update to simulate human experience
        # and be more computationally efficient; frames_per_state -1 because we take two steps per action
        if timestep % agent.frames_per_state - 1 == 0 or timestep == 0:
            frame = agent.preprocess(frame, frame2)
            frame[0] = timestep
            action = agent.act(env, frame)

            # Do two frames per action to take the max between frames and fix flickering issues
            frame_new, reward, done, _ = env.step(action)
            frame_new2, reward, done, _ = env.step(action)
            frame_new = agent.preprocess(frame_new, frame_new2)
            frame_new[0] = timestep

            memory.append((frame, action, reward, frame_new, done))

            frame = frame_new2
            frame2 = frame_new2

            # Train on the past transitions after filling memory with 50000 random examples
            if agent.random_init_counter >= agent.random_init:
                agent.fit_batch(memory)
            else:
                agent.random_init_counter += 1

            # Every few steps update target model to trained model to make it more likely to converge on the optimal Q*
            agent.C_counter += 1
            if agent.C_counter >= agent.C:
                agent.C_counter = 0
                agent.model_target = agent.copy_model(agent.model)
                agent.model_target.save('hlctdrl-model')
                print('model saved')

        else:
            frame, reward, done, _ = env.step(action)

        if done:
            env.render(close=True)
            break
