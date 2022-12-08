import sys
import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from arm_dynamics_student import ArmDynamicsStudent
from robot import Robot
from arm_gui import ArmGUI, Renderer
import argparse
import time
import math
import torch
np.set_printoptions(suppress=True)

class MPC:

    def __init__(self,):
        self.control_horizon = 10
        self.horizon = 30 #30
        self.alpha = 1 #1
        self.beta = 0 #0
        self.gamma = 1 #1
        self.du = .11 #.11

    # Cost function
    def compute_cost(self, goal, state, action, dynamics):
        L, T, V = dynamics.compute_energy(state)
        position = dynamics.compute_fk(state)

        cost = self.alpha * np.linalg.norm(goal-position) \
            - self.beta * np.linalg.norm(state[state.shape[0]-1,:]) \
            + self.gamma * T[0]
        return cost

    # Roll out a trajectory and compute its cost
    def roll_out(self, state, action_sequence, dynamics, goal):
        trajectory = np.tile(state, (self.horizon+1,1,1))
        for i in range(1,self.horizon+1):
            # Use dynamics to compute next state
            previous_state = trajectory[i-1]
            trajectory[i] = dynamics.advance(previous_state, action_sequence[i-1])
            
        cost = self.compute_cost(goal, trajectory[-1], action_sequence[0], dynamics)
        return cost
    
    def compute_action(self, dynamics, state, goal, action):
        ###### Add this ######
        #if isinstance(dynamics, ArmDynamicsTeacher):
        #    raise NotImplementedError  
        ###### End add ######
        
        #print("Goal is:")
        #print(goal)

        #print("State is:")
        #print(state)
        #print("Last action was:")
        #print(action)

        optimal_cost = 0
        change_in_cost = math.inf

        # While I am still improving
        while change_in_cost > .00001:
            #print("Improving action...")
            
            # Compute cost of current action sequence
            action_sequence = np.tile(action, (self.horizon,1,1))
            optimal_cost = self.roll_out(state, action_sequence, \
                                             dynamics, goal)
            #print("Cost of current action is: ", optimal_cost)
            
            # Perturb joints
            original_action = action.copy()
            for i in range(0, dynamics.num_links):
                new_action_add = original_action.copy()
                new_action_add[i,0] += self.du
                new_action_add_sequence = np.tile(new_action_add, \
                                                  (self.horizon,1,1))
                new_cost_add = self.roll_out(state, \
                                             new_action_add_sequence, \
                                             dynamics, goal)
                
                new_action_subtract = original_action.copy()
                new_action_subtract[i,0] -= self.du
                new_action_subtract_sequence = np.tile(new_action_subtract, \
                                                       (self.horizon,1,1))
                new_cost_subtract = self.roll_out(state, \
                                                  new_action_subtract_sequence, \
                                                  dynamics, goal)
                if new_cost_add < optimal_cost:
                    change_in_cost = optimal_cost - new_cost_add
                    optimal_cost = new_cost_add.copy()
                    action = new_action_add.copy()
                elif new_cost_subtract < optimal_cost:
                    change_in_cost = optimal_cost - new_cost_subtract
                    optimal_cost = new_cost_subtract.copy()
                    action = new_action_subtract.copy()
                else:
                    change_in_cost = 0
            #print("Cost of chosen action is: ", optimal_cost)
            #print("Change in cost was: ", change_in_cost)

        #print("Action is:")
        #print(action)
        return action


def main(args):

    # Arm
    arm = Robot(
        ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
    )

    # Dynamics model used for control
    if args.model_path is not None:
        dynamics = ArmDynamicsStudent(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
        dynamics.init_model(args.model_path, args.num_links, time_step=args.time_step, device=torch.device("cpu"))
    else:
        # Perfectly accurate model dynamics
        dynamics = ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )

    # Controller
    controller = MPC()

    # Control loop
    arm.reset()
    action = np.zeros((arm.dynamics.get_action_dim(), 1))
    goal = np.zeros((2, 1))
    goal[0, 0] = args.xgoal
    goal[1, 0] = args.ygoal
    arm.goal = goal

    if args.gui:
        renderer = Renderer()
        time.sleep(0.25)

    dt = args.time_step
    k = 0
    while True:
        t = time.time()
        arm.advance()
        if args.gui:
            renderer.plot([(arm, "tab:blue")])
        k += 1
        time.sleep(max(0, dt - (time.time() - t)))
        if k == controller.control_horizon:
            state = arm.get_state()
            action = controller.compute_action(dynamics, state, goal, action)
            arm.set_action(action)
            k = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_links", type=int, default=3)
    parser.add_argument("--link_mass", type=float, default=0.1)
    parser.add_argument("--link_length", type=float, default=1)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--time_limit", type=float, default=5)
    parser.add_argument("--gui", action="store_const", const=True, default=False)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--xgoal", type=float, default=-1.4)
    parser.add_argument("--ygoal", type=float, default=-1.4)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
