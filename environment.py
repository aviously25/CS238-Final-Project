from CARLO.agents import Car, Painting, RectangleBuilding
from CARLO.world import World
from CARLO.geometry import Point

import numpy as np
import random

class parkingSpot:
    def __init__(self, w: World, center: Point, direction: str):
        self.center = center
        self.direction = direction
        self.parkable = False
        self.heading = 0.0
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
            self.w.add(Painting(self.center, space_size, "red"))
        else: 
            self.w.add(Painting(self.center, space_size, "green"))
        
class environment:
    def __init__(self, w):
        self.parkingSpots = []
        self.target = None
        self.w = w

    def setUp(self):
        self.parkingSpots = [parkingSpot(self.w, point, direction) for point, direction in [
            (Point(35, 50), "up"), (Point(40, 50), "up"), (Point(45, 50), "up"), 
            (Point(25 ,30), "left"), (Point(25, 35), "left"), (Point(25, 40), "left"),
            (Point(55 ,30), "right"), (Point(55, 35), "right"), (Point(55, 40), "right")  
            ]
        ]
        selected = self.parkingSpots[random.randint(0, 8)] 
        selected.parkable = True
        self.target = selected

        for spot in self.parkingSpots: spot.make_spot()
       
        return selected
    
    def get_target(self):
        return self.target