import numpy as np

# try:
#     import pygame  # necessary only for the SteeringWheelController
# except ImportError:
#     print("pygame is not installed, you won't be able to use the steering wheel.")


class KeyboardController:
    def __init__(self, world):
        self._steering = 0.0
        self._throttle = 0.0

        self.min_steering = -0.5
        self.max_steering = +0.5

        self.min_throttle = -1.5
        self.max_throttle = +1.5

        world.visualizer.win.bind("<KeyRelease-Up>", self.arrow_up_release)
        world.visualizer.win.bind("<KeyRelease-Down>", self.arrow_down_release)
        world.visualizer.win.bind("<KeyRelease-Left>", self.arrow_left_release)
        world.visualizer.win.bind("<KeyRelease-Right>", self.arrow_right_release)

        world.visualizer.win.bind("<KeyPress-Up>", self.arrow_up_press)
        world.visualizer.win.bind("<KeyPress-Down>", self.arrow_down_press)
        world.visualizer.win.bind("<KeyPress-Left>", self.arrow_left_press)
        world.visualizer.win.bind("<KeyPress-Right>", self.arrow_right_press)

        world.visualizer.win.focus_force()  # very impolite... Polite version is focus_set(), but it is not always working

    @property
    def steering(self):
        return self._steering

    @property
    def throttle(self):
        return self._throttle

    @steering.setter
    def steering(self, val):
        self._steering = np.clip(val, self.min_steering, self.max_steering)

    @throttle.setter
    def throttle(self, val):
        self._throttle = np.clip(val, self.min_throttle, self.max_throttle)

    def arrow_up_release(self, event):
        self.throttle -= 1.5

    def arrow_down_release(self, event):
        self.throttle += 1.5

    def arrow_left_release(self, event):
        self.steering -= 0.5

    def arrow_right_release(self, event):
        self.steering += 0.5

    def arrow_up_press(self, event):
        self.throttle += 1.5

    def arrow_down_press(self, event):
        self.throttle -= 1.5

    def arrow_left_press(self, event):
        self.steering += 0.5

    def arrow_right_press(self, event):
        self.steering -= 0.5


class AutomatedController:
    def __init__(self):
        self._steering = 0.0
        self._throttle = 0.0

        self.min_steering = -0.5
        self.max_steering = +0.5

        self.min_throttle = -1.5
        self.max_throttle = +1.5

    @property
    def steering(self):
        return self._steering

    @property
    def throttle(self):
        return self._throttle

    @steering.setter
    def steering(self, val):
        self._steering = np.clip(val, self.min_steering, self.max_steering)

    @throttle.setter
    def throttle(self, val):
        self._throttle = np.clip(val, self.min_throttle, self.max_throttle)

    def do_action(self, action: int):
        if action == 0:
            pass
        if action == 1:  # speed up
            self.increase_throttle()
        elif action == 2:  # slow down
            self.decrease_throttle()
        elif action == 3:  # turn right
            self.steer_right()
        elif action == 4:  # turn left
            self.steer_left()

    def increase_throttle(self, val=5):
        self.throttle += val

    def decrease_throttle(self):
        self.throttle -= 1.5

    def steer_right(self):
        self.steering -= 0.25

    def steer_left(self):
        self.steering += 0.25


# class SteeringWheelController:  # For Logitech G29 Steering Wheel
#     def __init__(self, world):
#         pygame.init()
#         pygame.joystick.init()
#         self.joystick = pygame.joystick.Joystick(0)
#         self.joystick.init()
#
#     @property
#     def steering(self):
#         events = pygame.event.get()  # This is necessary
#         return -self.joystick.get_axis(0) / 2.0
#
#     @property
#     def throttle(self):
#         events = pygame.event.get()  # This is necessary
#         return -self.joystick.get_axis(1)
