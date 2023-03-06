from CARLO.agents import Car, Painting, RectangleBuilding
from CARLO.world import World
from CARLO.geometry import Point
from CARLO.interactive_controllers import AutomatedController, KeyboardController
from environment import environment, parkingSpot

import numpy as np
import time
import random
import sys

DT = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.


def scenario1(automated: bool = False):
    # add parking spots
    w = World(DT, width=80, height=60, bg_color="lightgray", ppm=16)
    env = environment(w)
    target = env.setUp()

    # add car
    c1 = Car(Point(40, 20), np.pi / 2, "blue")
    c1.max_speed = 5
    c1.min_speed = -2.5
    c1.set_control(0, 0)
    w.add(c1)

    # render initial world
    w.render()

    # create controller
    controller = AutomatedController(w) if automated else KeyboardController(w)

    while True:
        c1.set_control(controller.steering, controller.throttle)
        w.tick()
        # print(c1.get_offset(target.heading))
        w.render()

        # sleep so the car doesn't disappear from rendering too fast
        time.sleep(DT / 5)

        # simulate random action if automated
        if automated:
            random_action = random.choice([i for i in range(100)])
            controller.do_action(random_action)

        # check collision
        if c1.is_colliding(target):
            c1.obj.intersectPercent(target.spot.obj)
            print("Collision detected")


if __name__ == "__main__":
    automated = False

    if len(sys.argv) == 2:
        if sys.argv[1] == "--automated":
            automated = True

    scenario1(automated)
