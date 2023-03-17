from CARLO.agents import Car
from environment import environment as env
from CARLO.interactive_controllers import AutomatedController
from environment import environment

import numpy as np
from copy import deepcopy
from random import randrange, choice


class QLearning:
    def __init__(self, env: environment, num_sims=1000, num_episodes=100):
        self.env = env
        self.states_dim = [
            env.w.visualizer.display_width / 2,  # num of possible x positions
            env.w.visualizer.display_height / 2,  # num of possible y positions
            env.car.max_speed * 10,  # possible speed values (2 decimal places)
            int(2 * np.pi * 10),  # heading angle (2 decimal places)
        ]
        self.num_states = int(np.prod(self.states_dim))
        self.num_actions = 5
        self.discount_factor = 0.95
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.learning_rate = 0.01
        self.exploration_prob = 0.1

    def get_state(self):
        x = self.env.car.x
        y = self.env.car.y
        speed = self.env.car.speed
        heading = self.env.car.heading

        return np.ravel_multi_index(
            (x, y, speed, heading),
            dims=tuple(self.states_dim),
        )

    def simulate_action(self, action: int, car: Car, dt: float):
        c1 = Car(car.center, car.heading, "blue")
        cont = AutomatedController()

        c1.velocity = car.velocity
        c1.heading = car.heading
        c1.lr = car.rear_dist
        c1.lf = c1.lr
        c1.max_speed = 0.5
        c1.min_speed = -2.5
        c1.set_control(0, 0)

        cont.do_action(action)
        c1.set_control(cont.steering, cont.throttle)
        state = c1.simulate_next_state(dt)

        reward = self.env.reward_function(car=c1)

        return reward, c1

    def run_iter(self):
        dt = self.env.w.dt

        state = self.get_state()
        action = (
            np.argmax(self.q_table[state])
            if randrange(0, 1) > self.exploration_prob
            else choice([i for i in self.num_actions])
        )
        reward, c1 = self.simulate_action(action, self.env.car, dt)
        next_state = self.get_state()

        self.q_table[
            state, action
        ] += self.learning_rate * reward + self.discount_factor * max(
            self.q_table[next_state, :]
        )


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
        dt = self.env.w.dt*20

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
