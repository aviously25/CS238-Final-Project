from CARLO.agents import Car, Painting, RectangleBuilding
from CARLO.world import World
from CARLO.geometry import Point

from CARLO.interactive_controllers import KeyboardController

import numpy as np
import random

DT = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.

w = World(DT, width=80, height=60, bg_color="lightgray", ppm=16)

class parkingSpot:
    def __init__(self, center: Point, direction: str):
        self.center = center
        self.direction = direction
        self.parkable = False
        self.angle = 0.0

    def make_spot(self):
        space_size = None
        if(self.direction == "up"):
            self.angle = 0.0 # Make it face left so we can get the offset with car
            w.add(Painting(Point(self.center.x-2.5, self.center.y), Point(0.5, 7), "white"))
            w.add(Painting(Point(self.center.x+2.5, self.center.y), Point(0.5, 7), "white"))
            w.add(Painting(Point(self.center.x, self.center.y+3.5), Point(7, 0.5), "white"))
            space_size = Point(4, 6)
            
            
        elif(self.direction == "left"):
            w.add(Painting(Point(self.center.x, self.center.y-2.5), Point(7, 0.5), "white"))
            w.add(Painting(Point(self.center.x, self.center.y+2.5), Point(7, 0.5), "white"))
            w.add(Painting(Point(self.center.x-3.5, self.center.y), Point(0.5, 7), "white"))
            space_size = Point(6, 4) 

        elif(self.direction == "right"):
            w.add(Painting(Point(self.center.x, self.center.y-2.5), Point(7, 0.5), "white"))
            w.add(Painting(Point(self.center.x, self.center.y+2.5), Point(7, 0.5), "white"))
            w.add(Painting(Point(self.center.x+3.5, self.center.y), Point(0.5, 7), "white"))
            space_size = Point(6, 4)

        if(not self.parkable):
            w.add(Painting(self.center, space_size, "red"))
        else: 
            w.add(Painting(self.center, space_size, "green"))
        
class environment:
    def __init__(self):
        self.parkingSpots = []
        self.target = self.setUp()

    def setUp(self):
        self.parkingSpots = [parkingSpot(point, direction) for point, direction in [
            (Point(35, 50), "up"), (Point(40, 50), "up"), (Point(45, 50), "up"), 
            (Point(25 ,30), "left"), (Point(25, 35), "left"), (Point(25, 40), "left"),
            (Point(55 ,30), "right"), (Point(55, 35), "right"), (Point(55, 40), "right")  
            ]
        ]
        selected = self.parkingSpots[random.randint(0, 8)] 
        selected.parkable = True
        
        for spot in self.parkingSpots: spot.make_spot()

        return selected
    
    def get_target(self):
        return self.target

def scenario1():
    # add parking spots
    selected = environment()

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
        print(c1.center)
        w.render()

        if w.collision_exists():
            import sys

            sys.exit(0)


if __name__ == "__main__":
    scenario1()
