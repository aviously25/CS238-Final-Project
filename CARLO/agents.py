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
    
    def is_colliding(self, parking):
        return self.collidesWith(parking.spot)
    
    def car_park_distance(self, parking):
        return math.dist(self.center, parking.center)
    
class parkingSpot:
    def __init__(self, w, center: Point, direction: str):
        self.center = center
        self.direction = direction
        self.parkable = False
        self.heading = 0.0
        self.spot = None
        self.w = w

    def make_spot(self):
        space_size = None
        if(self.direction == "up"):
            self.angle = 0.0 # Make it face left so we can get the offset with car
            self.w.add(Painting(Point(self.center.x-2.5, self.center.y), Point(0.5, 7), "white"))
            self.w.add(Painting(Point(self.center.x+2.5, self.center.y), Point(0.5, 7), "white"))
            self.w.add(Painting(Point(self.center.x, self.center.y+3.5), Point(7, 0.5), "white"))
            space_size = Point(4, 6)
            self.heading=np.pi/2
            
        elif(self.direction == "left"):
            self.w.add(Painting(Point(self.center.x, self.center.y-2.5), Point(7, 0.5), "white"))
            self.w.add(Painting(Point(self.center.x, self.center.y+2.5), Point(7, 0.5), "white"))
            self.w.add(Painting(Point(self.center.x-3.5, self.center.y), Point(0.5, 7), "white"))
            space_size = Point(6, 4) 
            self.heading = np.pi

        elif(self.direction == "right"):
            self.w.add(Painting(Point(self.center.x, self.center.y-2.5), Point(7, 0.5), "white"))
            self.w.add(Painting(Point(self.center.x, self.center.y+2.5), Point(7, 0.5), "white"))
            self.w.add(Painting(Point(self.center.x+3.5, self.center.y), Point(0.5, 7), "white"))
            space_size = Point(6, 4)
            self.heading = 0

        if(not self.parkable):
            self.spot = Painting(self.center, space_size, "red")
        else: 
            self.spot = Painting(self.center, space_size, "green")
            
        self.w.add(self.spot)
        self.spot.collidable = True
        
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
