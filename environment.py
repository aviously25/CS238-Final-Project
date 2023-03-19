from CARLO.geometry import Point
from CARLO.agents import parkingSpot
import numpy as np
import random
from copy import deepcopy
from CARLO.world import World
from CARLO.agents import Car


class environment:
    def __init__(self, w: World):
        self.parkingSpots = []
        self.target = None
        self.car = None
        self.w = w
        self.controller = None

    def setUp(self, index=None):
        self.parkingSpots = [
            parkingSpot(self.w, point, direction)
            for point, direction in [
                (Point(10, 30), "up"),
                (Point(15, 30), "up"),
                (Point(20, 30), "up"),
                (Point(5, 10), "left"),
                (Point(5, 15), "left"),
                (Point(5, 20), "left"),
                (Point(25, 10), "right"),
                (Point(25, 15), "right"),
                (Point(25, 20), "right"),
            ]
        ]

        if index == None:
            index = random.randint(0, 8)
        self.park_index = index
        selected = self.parkingSpots[index]
        selected.parkable = True
        self.target = selected

        for spot in self.parkingSpots:
            spot.make_spot()

        return selected

    def get_target(self):
        return self.target

    def collide_non_target(self, car=None):
        if car is None:
            car = self.car

        for parking in self.parkingSpots:
            if parking is not self.target and car.is_colliding(parking):
                return True

        return False

    def reward_function(self, car):
        if car is None:
            car = self.car

        value = self.collide_non_target(car=car) * -10000
        value += car.park_dist(self.target, car=car) * (-100 * car.get_alive_time())
        value += car.get_offset(self.target.heading) * -200
        value += ((car.collisionPercent(self.target) * 100) ** 2) * 100
        # value += -100 * car.get_alive_time() ** 2

        return value
