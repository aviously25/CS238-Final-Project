from CARLO.geometry import Point
from CARLO.agents import parkingSpot
import numpy as np
import random
        
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