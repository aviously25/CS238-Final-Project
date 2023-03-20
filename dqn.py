from CARLO.agents import Car
from environment import environment
from CARLO.interactive_controllers import AutomatedController
from CARLO.world import World
from CARLO.geometry import Point

import numpy as np
from random import random, choice, randrange, sample
from tqdm import tqdm
import time

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# from threading import Thread

MAX_CAR_SPEED = 2.5
PPM_MODIFIER = 32
MAX_RUN_TIME = 7  # in seconds

BATCH_SIZE = 32


class DQN:
    def __init__(self, env: environment):
        self.env = env
        self.states_dim = [
            int(env.w.visualizer.display_width / PPM_MODIFIER)
            + 1,  # num of possible x positions
            int(env.w.visualizer.display_height / PPM_MODIFIER)
            + 1,  # num of possible y positions
            int(MAX_CAR_SPEED * 10) + 1,  # possible speed values (1 decimal places)
            int(2 * np.pi * 10) + 1,  # heading angle (1 decimal places)
        ]
        self.num_states = int(np.prod(self.states_dim))
        self.num_actions = 5
        self.discount = 0.95
        # self.q_table = np.full(
        #     (self.num_states, self.num_actions), -1000, dtype=np.float64
        # )
        self.learning_rate = 0.01
        self.exploration_prob = 1
        self.epsilon_start = 1
        self.epsilon_end = 0.001
        self.epsilon_decay = 0.995

        self.memory = deque(maxlen=2000)

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()

        model.add(Dense(4, input_dim=self.num_states, activation="relu"))
        model.add(Dense(2, activation="relu"))
        model.add(Dense(self.num_actions, activation="linear"))

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(lr=self.learning_rate),
        )

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random() < self.exploration_prob:
            return choice([i for i in range(self.num_actions)])

        act_vals = self.model.predict(state)
        return np.argmax(act_vals)

    def replay(self):
        minibatch = sample(self.memory, BATCH_SIZE)

        for s, a, r, sn, done in minibatch:
            target = r

            if not done:
                print(sn)
                target = r + self.discount * np.max(self.model.predict(sn))

            target_f = self.model.predict(s)
            target_f[0][a] = target

            self.model.fit(s, target_f, epoch=1, verbose=0)

        if self.exploration_prob > self.epsilon_end:
            self.exploration_prob *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    # the function will take the 4-parameter vector and turn it into 1 number
    def get_state(self, car):
        x = int(car.x * self.env.w.visualizer.ppm / PPM_MODIFIER)
        y = int(car.y * self.env.w.visualizer.ppm / PPM_MODIFIER)
        speed = int(car.speed * 10)
        heading = int(car.heading * 10)

        return (x, y, speed, heading)

        # try:
        #     return np.ravel_multi_index(
        #         (x, y, speed, heading),
        #         dims=tuple(self.states_dim),
        #     )
        # except Exception:
        #     return 0

    def run_iter(self, car: Car, controller: AutomatedController):
        dt = self.env.w.dt

        s = self.get_state(car)
        a = self.act(s)

        # print(
        #     f"best action: {best_action} with q_value {self.q_table[s][a]} out of {self.q_table[s]}"
        # )

        controller.do_action(a)
        car.set_control(controller.steering, controller.throttle)
        r = self.env.reward_function(car)
        # print(np.argmax(self.q_table[state]), self.q_table[state][action])
        # r, c1 = self.simulate_action(a, car, dt)

        car.tick(dt)
        sn = self.get_state(car)  # next state

        return (s, a, r, sn)

    def run_episode(self, env, w, car, controller):
        start_time = time.time()
        done = False
        while not done:
            s, a, r, sn = self.run_iter(car, controller)
            if (
                time.time() - start_time > MAX_RUN_TIME
                or car.check_bounds(w)
                or env.collide_non_target(car)
                or (car.collisionPercent(env.target) == 1 and car.speed < 0.1)
            ):
                done = True

            self.remember(s, a, r, sn, done)
            # controller.do_action(action)
            # car.set_control(controller.steering, controller.throttle)

            # w.render()
            # time.sleep(w.dt / 50)

            if done:
                final_reward = env.reward_function(car)
                return final_reward

            if len(self.memory) > BATCH_SIZE:
                self.replay()

    def train(
        self,
        env: environment,
        w: World,
        num_episodes: int = 10000,
    ):
        print(self.num_states)
        decay_factor = (self.epsilon_end / self.epsilon_start) ** (1 / num_episodes)
        for i in tqdm(range(num_episodes)):
            # initialize car
            # rand_x = randrange(10, 20)
            # rand_y = randrange(0, 20)
            car = Car(Point(15, 5), np.pi / 2, "blue")
            car.max_speed = MAX_CAR_SPEED
            car.set_control(0, 0)
            w.add(car)

            # initialize controller
            controller = AutomatedController()

            # w.render()

            # run the episode
            reward = self.run_episode(env, w, car, controller)
            if reward > 0:
                w.render()

            # remove car when done
            w.remove(car)

            self.exploration_prob = self.exploration_prob * decay_factor
            print(f"reward: {reward}, epsilon = {self.exploration_prob}")

            if i % 100 == 0 and i > 0:
                print("writing policy")
                self.save(f"dqn_models/dqn_model_{i}.hdf5")
            # self.write_policy()

        # self.write_policy()

    def write_policy(self):
        dest = "policy_" + str(self.env.park_index) + ".txt"
        with open(dest, "w") as f:
            policy = np.argmax(self.q_table, axis=1)
            for action in policy:  # for each state
                f.write(str(action) + "\n")  # action is the one with the highest reward

        print("done")
