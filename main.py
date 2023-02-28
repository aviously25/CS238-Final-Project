from CARLO.agents import Car, Painting, RectangleBuilding
from CARLO.world import World
from CARLO.geometry import Point
import numpy as np

DT = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.


def main():
    w = World(DT, width=80, height=60, bg_color="lightgray", ppm=16)

    # add parking spot
    w.add(Painting(Point(40, 50), Point(4, 6), "green"))  # We build a sidewalk.

    # add collision area
    # w.add(
    #     RectangleBuilding(Point(72.5, 107.5), Point(95, 25))
    # )  # The RectangleBuilding is then on top of the sidewalk, with some margin.

    # add car
    c1 = Car(Point(40, 20), np.pi / 2, "blue")
    w.add(c1)

    while True:
        w.render()


if __name__ == "__main__":
    main()
