from CARLO.agents import Car, Painting, RectangleBuilding
from CARLO.world import World
from CARLO.geometry import Point

from CARLO.interactive_controllers import KeyboardController
from environment import environment, parkingSpot
import numpy as np

DT = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.

def scenario1():
    # add parking spots
    w = World(DT, width=80, height=60, bg_color="lightgray", ppm=16)
    env = environment(w)
    target = env.setUp()

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
        print(c1.get_offset(target.heading))
        w.render()

        if w.collision_exists():
            import sys

            sys.exit(0)


if __name__ == "__main__":
    scenario1()
