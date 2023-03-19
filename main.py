from CARLO.agents import Car, Painting, RectangleBuilding
from CARLO.world import World
from CARLO.geometry import Point
from CARLO.interactive_controllers import AutomatedController, KeyboardController
from environment import environment, parkingSpot
from algorithms import ForwardSearch, QLearning
import numpy as np
import time
import random
import sys
import cProfile
import pstats

DT = 0.5  # time steps in terms of seconds. In other words, 1/dt is the FPS.
SPOT_NUM = 3
PPM = 8
WIDTH = 30
HEIGHT = 40


def run_policy(file):
    print("Running policy")

    q_table = np.loadtxt(file)

    w = World(DT, width=WIDTH, height=HEIGHT, bg_color="lightgray", ppm=PPM)
    env = environment(w)
    env.setUp(SPOT_NUM)

    # initialize car
    car = Car(Point(15, 5), np.pi / 2, "blue")
    car.max_speed = 2.5
    car.min_speed = -2.5
    car.set_control(0, 0)
    w.add(car)

    # initialize controller
    controller = AutomatedController()

    w.render()

    # init variables for running an episode
    start_time = time.time()
    run_sim = True
    q = QLearning(env)
    # run the episode
    while run_sim:
        car.set_control(controller.steering, controller.throttle)

        action = q_table[q.get_state(car)]
        controller.do_action(action)

        w.tick()
        w.render()

        # sleep so the car doesn't disappear from rendering too fast
        time.sleep(w.dt / 5)

        if time.time() - start_time > 20 or q.get_state(car) < 0:
            run_sim = False

    # remove car when done
    w.remove(car)


def q_learning(automated: bool = False):
    print("Running q")
    w = World(DT, width=WIDTH, height=HEIGHT, bg_color="lightgray", ppm=PPM)
    env = environment(w)
    target = env.setUp(SPOT_NUM)

    Q = QLearning(env)

    Q.train(env, w, num_episodes=5000)


def forwardSearch(automated: bool = True):
    print("Running forward")
    # add parking spots
    w = World(DT, width=30, height=40, bg_color="lightgray", ppm=8)
    env = environment(w)
    target = env.setUp(3)

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
    p.strip_dirs().sort_stats("tottime").print_stats()


if __name__ == "__main__":
    task = "start"

    if len(sys.argv) >= 2:
        task = sys.argv[1]

    if task == "q":
        cProfile.run("q_learning()", "stats.raw")
    elif task == "f":
        forwardSearch()
    elif task == "p":
        file = sys.argv[2]
        run_policy(file)
    elif task == "s":
        show_stats()
    else:
        forwardSearch(False)
