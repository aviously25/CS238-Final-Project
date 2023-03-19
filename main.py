from CARLO.agents import Car, Painting, RectangleBuilding
from CARLO.world import World
from CARLO.geometry import Point
from CARLO.interactive_controllers import AutomatedController, KeyboardController
from environment import environment, parkingSpot
from algorithms import ForwardSearch, QLearning
from dqn import DQN
import numpy as np
import time
import random
import sys
import cProfile
import pstats
import shutil

DT = 0.5  # time steps in terms of seconds. In other words, 1/dt is the FPS.
SPOT_NUM = 4
PPM = 8
WIDTH = 30
HEIGHT = 40
MAX_TIME = 5


def run_all_cont():
    best_rewards = [-2000 for i in range(9)]

    while True:
        # for spot num
        for i in range(9):
            print(f"Running Q Learning on spot {i}")
            q_learning(i)
            file = f"policies/policy_{i}.txt"
            reward = run_policy(i, render=False)

            if reward > best_rewards[i]:
                print("best reward:", reward)
                shutil.move(file, f"policy_{i}_BEST.txt")
                best_rewards[i] = reward


def dqn_learning(spot=SPOT_NUM):
    w = World(DT, width=WIDTH, height=HEIGHT, bg_color="lightgray", ppm=PPM)
    env = environment(w)
    target = env.setUp(spot)

    dqn = DQN(env)

    dqn.train(env, w, num_episodes=50000)


def run_policy(spot=SPOT_NUM, render=True):
    # print("Running policy")

    file = f"policies/policy_{spot}.txt"
    q_table = np.loadtxt(file)

    w = World(DT, width=WIDTH, height=HEIGHT, bg_color="lightgray", ppm=PPM)
    env = environment(w)
    env.setUp(spot)

    # initialize car
    car = Car(Point(15, 5), np.pi / 2, "blue")
    car.max_speed = 2.5
    car.min_speed = 0
    car.set_control(0, 0)
    w.add(car)

    # initialize controller
    controller = AutomatedController()

    if render:
        w.render()

    # init variables for running an episode
    q = QLearning(env)
    start_time = time.time()
    final_reward = 0

    # run the episode
    while True:
        car.set_control(controller.steering, controller.throttle)

        action = q_table[q.get_state(car)]
        controller.do_action(action)

        w.tick()

        if render:
            w.render()
            # sleep so the car doesn't disappear from rendering too fast
            time.sleep(w.dt / 5)

        if (
            time.time() - start_time > MAX_TIME
            # or car.check_bounds(w)
            or env.collide_non_target(car)
            or (car.collisionPercent(env.target) == 1 and car.speed < 0.1)
        ):
            final_reward = env.reward_function(car)
            break

    # remove car when done
    print("final reward: ", final_reward)
    return final_reward


def q_learning(spot=SPOT_NUM, automated: bool = False):
    w = World(DT, width=WIDTH, height=HEIGHT, bg_color="lightgray", ppm=PPM)
    env = environment(w)
    target = env.setUp(spot)

    Q = QLearning(env)

    Q.train(env, w, num_episodes=100000)


def forwardSearch(automated: bool = True):
    print("Running forward")
    # add parking spots
    w = World(DT, width=30, height=40, bg_color="lightgray", ppm=8)
    env = environment(w)
    target = env.setUp(SPOT_NUM)

    # add car
    car = Car(Point(15, 3), np.pi / 2, "blue")
    env.car = car
    car.max_speed = 2.5
    car.min_speed = -2.5
    car.set_control(0, 0)
    w.add(car)

    # render initial world
    w.render()

    # create controller
    controller = AutomatedController() if automated else KeyboardController(w)
    env.controller = controller

    fs = ForwardSearch(env)

    while True:
        car.set_control(controller.steering, controller.throttle)
        w.tick()
        # print(c1.get_offset(target.heading))
        w.render()

        # sleep so the car doesn't disappear from rendering too fast
        time.sleep(DT / 5)

        # simulate random action if automated
        if automated:
            if env.car.park_dist(target) > 2:
                reward, best_action = fs.run_iter(env.car, 2)
                controller.do_action(best_action)
                print(best_action)
            else:
                reward, best_action = fs.run_iter(env.car, 1)
                controller.do_action(best_action)

            if env.collide_non_target(car):
                print("car crashed")
                time.sleep(3)
                sys.exit(0)

            print(reward)
            print(best_action)

        # check collision
        if car.is_colliding(target):
            print(car.collisionPercent(target))
        print(env.reward_function(car))

        if (
            # or car.check_bounds(w)
            env.collide_non_target(car)
            or (car.collisionPercent(env.target) == 1 and car.speed < 0.1)
        ):
            final_reward = env.reward_function(car)
            return


def show_stats():
    p = pstats.Stats("stats.raw")
    p.strip_dirs().sort_stats("cumtime").print_stats()


if __name__ == "__main__":
    task = ""

    if len(sys.argv) >= 2:
        task = sys.argv[1]

    if task == "q":
        if len(sys.argv) > 2:
            spot = int(sys.argv[2])
            q_learning(spot)
        else:
            q_learning()
    elif task == "d":
        if len(sys.argv) > 2:
            spot = int(sys.argv[2])
            dqn_learning(spot)
        else:
            dqn_learning()
    elif task == "f":
        forwardSearch()
    elif task == "p":
        if len(sys.argv) > 2:
            spot = int(sys.argv[2])
            run_policy(spot)
        else:
            run_policy()
    elif task == "s":
        show_stats()
    elif task == "a":
        run_all_cont()
    else:
        forwardSearch(False)
