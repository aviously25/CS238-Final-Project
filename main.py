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

DT = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.


def q_learning(automated: bool = False):
    w = World(DT, width=30, height=40, bg_color="lightgray", ppm=16)
    env = environment(w)
    target = env.setUp()

    Q = QLearning(env)

    Q.train(env, w)


def forwardSearch(automated: bool = True):
    # add parking spots
    w = World(DT, width=30, height=40, bg_color="lightgray", ppm=16)
    env = environment(w)
    target = env.setUp(3)

    # add car
    c1 = Car(Point(15, 3), np.pi / 2, "blue")
    env.car = c1
    c1.max_speed = 1
    c1.min_speed = -2.5
    c1.set_control(0, 0)
    w.add(c1)

    # render initial world
    w.render()

    # create controller
    controller = AutomatedController() if automated else KeyboardController(w)
    env.controller = controller

    fs = ForwardSearch(env)

    while True:
        c1.set_control(controller.steering, controller.throttle)
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

            if env.collide_non_target(c1):
                print("car crashed")
                time.sleep(3)
                sys.exit(0)

            print(reward)
            print(best_action)

        # check collision
        if c1.is_colliding(target):
            print(c1.collisionPercent(target))
        print(env.reward_function(c1))


if __name__ == "__main__":
    task = "q"

    if len(sys.argv) == 2:
        task = sys.argv[1]

    if task == "q":
        q_learning()
    elif task == "f":
        forwardSearch()
    else:
        forwardSearch(False)
