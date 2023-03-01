from CARLO.entities import RectangleEntity, CircleEntity, RingEntity
from CARLO.geometry import Point

from math import atan
import numpy as np

# For colors, we use tkinter colors. See http://www.science.smith.edu/dftwiki/index.php/Color_Charts_for_TKinter


class Car(RectangleEntity):
    def __init__(self, center: Point, heading: float, color: str = "red"):
        size = Point(4.0, 2.0)
        movable = True
        friction = 0.06
        super(Car, self).__init__(center, heading, size, movable, friction)
        self.color = color
        self.collidable = True

    def get_offset(self, target: float):
        x = target
        y = self.heading
        ret = np.arctan2(np.sin(x-y), np.cos(x-y))
        #val = (ret * 180) / np.pi if you ever want to turn rads to degs
        return abs(ret)


class Pedestrian(CircleEntity):
    def __init__(
        self, center: Point, heading: float, color: str = "LightSalmon3"
    ):  # after careful consideration, I decided my color is the same as a salmon, so here we go.
        radius = 0.5
        movable = True
        friction = 0.2
        super(Pedestrian, self).__init__(center, heading, radius, movable, friction)
        self.color = color
        self.collidable = True


class RectangleBuilding(RectangleEntity):
    def __init__(self, center: Point, size: Point, color: str = "gray26"):
        heading = 0.0
        movable = False
        friction = 0.0
        super(RectangleBuilding, self).__init__(
            center, heading, size, movable, friction
        )
        self.color = color
        self.collidable = True


class CircleBuilding(CircleEntity):
    def __init__(self, center: Point, radius: float, color: str = "gray26"):
        heading = 0.0
        movable = False
        friction = 0.0
        super(CircleBuilding, self).__init__(center, heading, radius, movable, friction)
        self.color = color
        self.collidable = True


class RingBuilding(RingEntity):
    def __init__(
        self,
        center: Point,
        inner_radius: float,
        outer_radius: float,
        color: str = "gray26",
    ):
        heading = 0.0
        movable = False
        friction = 0.0
        super(RingBuilding, self).__init__(
            center, heading, inner_radius, outer_radius, movable, friction
        )
        self.color = color
        self.collidable = True


class Painting(RectangleEntity):
    def __init__(
        self, center: Point, size: Point, color: str = "gray26", heading: float = 0.0
    ):
        movable = False
        friction = 0.0
        super(Painting, self).__init__(center, heading, size, movable, friction)
        self.color = color
        self.collidable = False
