from CARLO.agents import Car
from environment import environment
from CARLO.interactive_controllers import AutomatedController
from CARLO.world import World
from CARLO.geometry import Point

import numpy as np
from copy import deepcopy
from random import randrange, choice
from tqdm import tqdm
import time

MAX_CAR_SPEED = 2.5
PPM_MODIFIER = 10


class QLearning:
    def __init__(self, env: environment, num_sims=1000, num_episodes=100):
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
        self.discount_factor = 0.95
        self.q_table = np.full((self.num_states, self.num_actions), -10000)
        self.learning_rate = 2
        self.exploration_prob = 0.3

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
            return -1

    def simulate_action(self, action: int, car: Car, dt: float):
        test_car = Car(car.center, car.heading, "blue")
        controller = AutomatedController()

        test_car.velocity = car.velocity
        test_car.heading = car.heading
        test_car.lr = car.rear_dist
        test_car.lf = test_car.lr
        test_car.max_speed = MAX_CAR_SPEED
        test_car.min_speed = -2.5
        test_car.set_control(0, 0)

        controller.do_action(action)
        test_car.set_control(controller.steering, controller.throttle)
        state = test_car.simulate_next_state(dt)

        reward = self.env.reward_function(car=test_car)

        return reward, test_car

    def run_iter(self, car: Car):
        dt = self.env.w.dt

        state = self.get_state(car)
        action = (
            np.argmax(self.q_table[state])
            if randrange(0, 1) < self.exploration_prob
            else choice([i for i in range(self.num_actions)])
        )
        print(np.argmax(self.q_table[state]), self.q_table[state][action])
        reward, c1 = self.simulate_action(action, car, dt)
        next_state = self.get_state(car)

        self.q_table[
            state, action
        ] += self.learning_rate * reward + self.discount_factor * max(
            self.q_table[next_state, :]
        )

        return action

    def train(
        self, env: environment, w: World, num_episodes: int = 1000, num_threads: int = 5
    ):
        for i in range(num_episodes):
            # initialize car
            car = Car(Point(15, 5), np.pi / 2, "blue")
            car.max_speed = MAX_CAR_SPEED
            car.min_speed = -2.5
            car.set_control(0, 0)
            w.add(car)

            # initialize controller
            controller = AutomatedController()

            w.render()

            # init variables for running an episode
            start_time = time.time()
            run_sim = True

            # run the episode
            while run_sim:
                car.set_control(controller.steering, controller.throttle)

                action = self.run_iter(car)
                controller.do_action(action)

                w.tick()
                w.render()

                # sleep so the car doesn't disappear from rendering too fast
                time.sleep(w.dt / 5)

                if time.time() - start_time > 20 or self.get_state(car) < 0:
                    run_sim = False

            # remove car when done
            w.remove(car)
    
    def write_policy(self):
        dest = "policy_"+str(self.env.park_index)+ ".txt"
        with open(dest, 'w') as f:
            for s in range(len(self.q_table)): # for each state
                #f.write(str(np.argmax(self.q_table[s]))+'\n') # action is the one with the highest reward
                f.write("1\n")
                if s %100000 == 0:
                    print(s)

        print("done")


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
