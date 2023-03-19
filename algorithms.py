from CARLO.agents import Car
from environment import environment
from CARLO.interactive_controllers import AutomatedController
from CARLO.world import World
from CARLO.geometry import Point

import numpy as np
from copy import deepcopy
from random import random, choice, randrange
from tqdm import tqdm
import time

# from threading import Thread

MAX_CAR_SPEED = 2.5
PPM_MODIFIER = 32
MAX_RUN_TIME = 7  # in seconds


class QLearning:
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
        self.q_table = np.full(
            (self.num_states, self.num_actions), -1000, dtype=np.float
        )
        self.learning_rate = 0.01
        self.exploration_prob = 1
        self.epsilon_start = 1
        self.epsilon_end = 0.001

    # the function will take the 4-parameter vector and turn it into 1 number
    def get_state(self, car):
        x = int(car.x * self.env.w.visualizer.ppm / PPM_MODIFIER)
        y = int(car.y * self.env.w.visualizer.ppm / PPM_MODIFIER)
        speed = int(car.speed * 10)
        heading = int(car.heading * 10)

        try:
            return np.ravel_multi_index(
                (x, y, speed, heading),
                dims=tuple(self.states_dim),
            )
        except Exception:
            return 0

    def run_iter(self, car: Car, controller: AutomatedController):
        dt = self.env.w.dt

        s = self.get_state(car)
        best_action = np.argmax(self.q_table[s])
        a = 0
        if random() > self.exploration_prob:
            a = best_action
        else:
            a = choice([i for i in range(self.num_actions)])

        # print(
        #     f"best action: {best_action} with q_value {self.q_table[s][a]} out of {self.q_table[s]}"
        # )

        controller.do_action(a)
        car.set_control(controller.steering, controller.throttle)
        r = self.env.reward_function(car)
        # print(np.argmax(self.q_table[state]), self.q_table[state][action])
        # r, c1 = self.simulate_action(a, car, dt)
        sn = self.get_state(car)  # next state

        car.tick(dt)

        try:
            new_q_value = (1 - self.learning_rate) * self.q_table[
                s, a
            ] + self.learning_rate * (
                r + self.discount * np.max(self.q_table[sn]) - self.q_table[s, a]
            )

            self.q_table[s, a] = new_q_value

            # print(f"q_table[{s}, {a}] = {new_q_value}")
        except Exception as err:
            print(err.args)

        return a

    def run_episode(self, env, w, car, controller):
        start_time = time.time()
        while True:
            self.run_iter(car, controller)
            # controller.do_action(action)
            # car.set_control(controller.steering, controller.throttle)

            # w.render()
            # time.sleep(w.dt / 50)

            if (
                time.time() - start_time > MAX_RUN_TIME
                # or car.check_bounds(w)
                or env.collide_non_target(car)
                or (car.collisionPercent(env.target) == 1 and car.speed < 0.1)
            ):
                print("time taken: ", time.time() - start_time)
                final_reward = env.reward_function(car)
                return

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
            rand_x = randrange(10, 20)
            rand_y = randrange(0, 20)
            car = Car(Point(rand_x, rand_y), np.pi / 2, "blue")
            car.max_speed = MAX_CAR_SPEED
            car.set_control(0, 0)
            w.add(car)

            # initialize controller
            controller = AutomatedController()

            # w.render()

            # run the episode
            self.run_episode(env, w, car, controller)

            # remove car when done
            w.remove(car)

            # self.exploration_prob = self.exploration_prob * decay_factor

            if i % 1000 == 0 and i > 0:
                print("writing policy")
                self.write_policy()

        self.write_policy()

    def write_policy(self):
        dest = "policy_" + str(self.env.park_index) + ".txt"
        with open(dest, "w") as f:
            policy = np.argmax(self.q_table, axis=1)
            for action in policy:  # for each state
                f.write(str(action) + "\n")  # action is the one with the highest reward

        print("done")

    # def train_threads(
    #     self,
    #     env: environment,
    #     w: World,
    #     num_episodes: int = 1000,
    #     num_threads: int = 20,
    # ):
    #     print(self.num_states)
    #     for i in tqdm(range(num_episodes)):
    #         # init variables for running an episode
    #         start_time = time.time()
    #
    #         cars = []
    #         controllers = []
    #         threads = []
    #         for i in range(num_threads):
    #             cars.append(Car(Point(15, 5), np.pi / 2, "blue"))
    #             cars[i].max_speed = MAX_CAR_SPEED
    #             cars[i].min_speed = -2.5
    #             cars[i].set_control(0, 0)
    #             w.add(cars[i])
    #             controllers.append(AutomatedController())
    #             threads.append(
    #                 Thread(
    #                     target=self.run_episode,
    #                     args=(env, cars[i], controllers[i], start_time, w),
    #                 )
    #             )
    #
    #         # run the episodes
    #         for thread in threads:
    #             thread.start()
    #
    #         for thread in threads:
    #             thread.join()
    #
    #         w.render()
    #
    #         # sleep so the car doesn't disappear from rendering too fast
    #         # time.sleep(w.dt / 10)
    #
    #         # remove car when done
    #         for car in cars:
    #             w.remove(car)
    #
    #     # write policy when all iters are done
    #     self.write_policy()


class ForwardSearch:
    # Why does forward search fail?
    # Its because it seeks out the highest reward which isnt always
    # in our best interest. For example, it will get really close to
    # a parking spot because that would maximize reward (distance)
    # , which we dont want cause it will crash.
    def __init__(self, env: environment):
        self.env = env

    def simulate_action(self, action: int, car: Car, dt: float):
        c1 = Car(car.center, car.heading, "blue")
        cont = AutomatedController()

        c1.velocity = car.velocity
        c1.heading = car.heading
        c1.lr = car.rear_dist
        c1.lf = c1.lr
        c1.max_speed = 0.5
        c1.min_speed = -2.5
        c1.angular_velocity = car.angular_velocity
        c1.set_control(0, 0)

        cont.do_action(action)
        c1.set_control(cont.steering, cont.throttle)
        state = c1.simulate_next_state(dt)

        reward = self.env.reward_function(car=c1)

        return reward, c1

    def run_iter(self, car: Car, iter):

        # make 5 copies of self.car
        dt = self.env.w.dt * 20

        none, n_c = self.simulate_action(0, car, dt)
        speed, n_s = self.simulate_action(1, car, dt)
        slow, n_d = self.simulate_action(2, car, dt)
        right, n_r = self.simulate_action(3, car, dt)
        left, n_l = self.simulate_action(4, car, dt)

        if iter == 2:
            iter = iter - 1
            none2, action_n = self.run_iter(n_c, iter)
            speed2, action_sp = self.run_iter(n_s, iter)
            slow2, action_s = self.run_iter(n_d, iter)
            right2, action_r = self.run_iter(n_r, iter)
            left2, action_l = self.run_iter(n_l, iter)

            sims = (none2, speed2, slow2, right2, left2)
            best = np.argmax(sims)
            print(
                f"None2: {none2}, Speed2:{speed2}, Slow2: {slow2}, Right2:{right2}, Left2:{left2}"
            )
            return sims[best], best

        print(f"None: {none}, Speed:{speed}, Slow: {slow}, Right:{right}, Left:{left}")
        sims = (none, speed, slow, right, left)
        best = np.argmax(sims)

        return sims[best], best
