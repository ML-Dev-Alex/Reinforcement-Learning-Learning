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
LOAD = False

EPISODES = 1000000

# SGD updates are sampled from this number of most recent frames
MEMORY = 10000000

# 5 minutes running on 60 FPS
MAX_TIME = 18000

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

        # Image Pre-processing, pairs of actions because we take two steps every time we act to fix atari flickering
        self.pairs_per_state = 2
        self.frame_count = 0

        self.batch_size = 32

        # Walk this many random steps before sampling batch for better exploration
        if LOAD:
            self.random_init = self.batch_size * 4
        else:
            self.random_init = 50000

        self.random_init_counter = 0


        # Target network update Frequency
        self.C = 10000
        self.C_counter = 0

        # Just for debugging
        self.debug_reward = 0

        if LOAD:
            self.model = keras.models.load_model('./save/hlctdrl-model', custom_objects={'huber_loss': self.huber_loss})
        else:
            self.model = self.build_model(state_size, action_size)

        self.model_target = self.copy_model(self.model)

    def build_model(self, state_size, action_size):
        frames_input = Input(state_size, name='frames')
        actions_input = Input((action_size,), name='mask')

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
        # Act randomly for a while to build some memory
        if agent.random_init_counter < agent.random_init:
            act_value = np.random.randint(0, env.action_space.n, size=1)[0]
        else:
            # Explore less randomly over time
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon_decay * self.epsilon

            # Take a random exploratory action sometimes, otherwise take best action it can
            if random.random() < self.epsilon:
                act_value = np.random.randint(0, env.action_space.n, size=1)[0]
            else:
                # Use ones as a mask to ensure we are selecting the best learned parameters
                ones = np.ones(action_size)
                # Make sure to fit the shape required by the model ( 1 hot encoded vector of action values )
                ones = np.reshape(ones, [1, action_size])
                act_value = np.argmax(self.model.predict([frame, ones], batch_size=1), axis=1)

        return act_value

    def fit_batch(self, memory):
        minibatch = random.sample(memory, self.batch_size)
        for i in range(self.batch_size):
            frame = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            frame_new = minibatch[i][3]
            done = minibatch[i][4]

            # Make sure action fits in the model
            action = np.reshape(action, [1, action_size])
            if done:
                target = reward
            else:
                # Pass ones as a mask to have the agent select the best action based on target model
                ones = np.ones_like(action)
                action_next = self.model_target.predict([frame_new, ones], batch_size=1)
                # Try to learn from working backwards from best state to start state
                # to find a clear path from action to reward, even if separated by many frames
                # Also known as Bellman's equation
                target = reward + self.gamma * np.argmax(action_next, axis=1)

            # Fit the model with the squared difference between the best predicted action (from the model_target)
            # and the action taken by the current model, this is done to try to help the current model get better at
            # predicting what the best action chosen by the model_target, such that we separate the tasks of learning
            # how to chose the best value from the Q function and actually updating the Q function itself
            y = action * target
            # Just a way to visualize progress while training
            agent.debug_reward += reward
            self.model.fit([frame, action], y, batch_size=1, verbose=0)

    # Consistent way to copy models with any keras version
    # Also a good opportunity to save the model since it only happens every so often
    def copy_model(self, model):
        model.save('./save/hlctdrl-model')
        print('model saved')
        return keras.models.load_model('./save/hlctdrl-model', custom_objects={'huber_loss': self.huber_loss})

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

    # Huber loss uses squared error when difference is big and normal distance when its large
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
    act_value = np.random.randint(0, env.action_space.n, size=1)[0]
    agent.debug_reward = 0

    if agent.C_counter >= agent.C:
        agent.C_counter = 0
        agent.model_target = agent.copy_model(agent.model)

    for timestep in range(MAX_TIME):
        # Render the last iteration and every time a model is saved/the target model gets updated
        # if episode == (EPISODES - 1) or agent.C_counter >= agent.C:
        #     env.render()

        # Increase the counter every frame, to be able to start learning once we have enough remembered states
        if agent.random_init_counter <= agent.random_init:
            agent.random_init_counter += 1

            # Only start saving model after initial random memory is saved
            # Only update the target model every 'C' steps to try to find the optimal Q* with more reliability

        agent.C_counter += 1


        # Repeat same action a few frames before next update to simulate human experience
        # and be more computationally efficient;
        # Divided by 2 because we take two steps per repeated action and training action
        if (timestep % agent.pairs_per_state == 0) or timestep == 0:
            # Evaluate them as one to fix flickering issues
            frame = agent.preprocess(frame, frame2)

            # Act
            action = np.zeros(action_size)
            act_value = agent.act(env, frame)
            # Action is a one hot encoded vector with zeroes on all entries except the best possible value
            action[act_value] = 1
            # Take another two steps to complete full action transition
            frame_new, _, _, _ = env.step(act_value)
            frame_new2, reward, done, _ = env.step(act_value)
            # Save them as 1 to fix flickering issues again
            frame_new = agent.preprocess(frame_new, frame_new2)

            # Enumerate frames passed to the model for the Conv2D layer, should be 1 frame for every (n = 4) env steps
            agent.frame_count += 1
            frame[0] = agent.frame_count
            frame_new[0] = agent.frame_count

            # Record transition for batch training
            memory.append((frame, action, reward, frame_new, done))

            # In case we don't repeat any action
            frame = frame_new
            frame2 = frame_new2

            # Train on the past transitions after filling memory with 'random_init'random examples
            if agent.random_init_counter >= agent.random_init:
                agent.fit_batch(memory)

        # Repeat same action for some steps
        else:
            # You could take individual steps here, but the code would become more complicated for almost no reason
            frame, _, _, _ = env.step(act_value)
            frame2, reward, done, _ = env.step(act_value)

        if done:
            # Break out of episode early in case of game over and make sure to close all open render windows
            env.render(close=True)
            break
