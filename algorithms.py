from CARLO.agents import Car
from environment import environment as env
from CARLO.interactive_controllers import AutomatedController
from environment import environment

import numpy as np
from copy import deepcopy

class forwardSearch:
    def __init__(self, env: environment):
        self.env = env

    def simulate_action(self, action: int, car: Car, dt: float):
        c1 = Car(car.center, car.heading, "blue")
        cont = AutomatedController()

        c1.velocity = car.velocity
        c1.heading = car.heading
        c1.lr = car.rear_dist
        c1.lf = c1.lr
        c1.max_speed = 0.5
        c1.min_speed = -2.5
        c1.set_control(0, 0) 

        cont.do_action(action)
        c1.set_control(cont.steering, cont.throttle)
        state = c1.simulate_next_state(dt)
        
        reward = self.env.reward_function(car=c1)
        
        return reward, c1
        

    def run_iter(self, car: Car, iter):

        # make 5 copies of self.car
        dt = self.env.w.dt * 60

        none, n_c = self.simulate_action(0, car, dt)
        speed, n_s = self.simulate_action(1, car, dt)
        slow, n_d = self.simulate_action(2, car, dt)
        right, n_r = self.simulate_action(3, car, dt)
        left, n_l = self.simulate_action(4, car, dt)

        if(iter == 2):
            iter = iter -1
            none2, action_n = self.run_iter(n_c, iter) 
            speed2, action_sp = self.run_iter(n_s, iter)
            slow2, action_s = self.run_iter(n_d, iter) 
            right2, action_r = self.run_iter(n_r, iter)
            left2, action_l = self.run_iter(n_l, iter)

            sims = (none2, speed2, slow2, right2, left2)
            best = np.argmax(sims)
            print(f"None2: {none2}, Speed2:{speed2}, Slow2: {slow2}, Right2:{right2}, Left2:{left2}")
            return sims[best], best
            
 
        
        print(f"None: {none}, Speed:{speed}, Slow: {slow}, Right:{right}, Left:{left}")
        sims = (none, speed, slow, right, left)
        best = np.argmax(sims)
        
        return sims[best], best
    

    
    