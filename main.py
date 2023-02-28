from CARLO.agents import Car, Painting, RectangleBuilding
from CARLO.world import World
from CARLO.geometry import Point

from CARLO.interactive_controllers import KeyboardController

import numpy as np

DT = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.


def scenario1():
    w = World(DT, width=80, height=60, bg_color="lightgray", ppm=16)

    # add parking lines
    w.add(Painting(Point(40, 53.25), Point(16, 0.5), "white"))
    w.add(Painting(Point(37.75, 50), Point(0.5, 7), "white"))
    w.add(Painting(Point(42.25, 50), Point(0.5, 7), "white"))
    w.add(Painting(Point(33.75, 50), Point(0.5, 7), "white"))
    w.add(Painting(Point(46.25, 50), Point(0.5, 7), "white"))

    # add parking spot
    w.add(Painting(Point(40, 50), Point(4, 6), "green"))

    # add car
    c1 = Car(Point(40, 20), np.pi / 2, "blue")
    c1.max_speed = 10
    c1.min_speed = -5
    c1.set_control(0, 0)
    w.add(c1)

    # render initial world
    w.render()

    controller = KeyboardController(w)
    while True:
        c1.set_control(controller.steering, controller.throttle)
        w.tick()
        print(c1.velocity)
        w.render()

        if w.collision_exists():
            import sys

            sys.exit(0)


if __name__ == "__main__":
    scenario1()
