import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
import argparse
import os
import random
import ray

np.set_printoptions(suppress=True)

# Seeding for reproducibility
random.seed(0)
np.random.seed(0)

# ray
#ray.init()

# define number of samples
sample_number = 900000

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=0.01)
    parser.add_argument('--time_limit', type=float, default=5)
    parser.add_argument('--save_dir', type=str, default='dataset')
    return parser.parse_args()
"""
@ray.remote
def generate_x(arm_teacher):    
    sample = np.zeros((arm_teacher.dynamics.get_state_dim() + arm_teacher.dynamics.get_action_dim(),1))
    for link in range(0, arm_teacher.dynamics.get_num_links()):
        sample[link,0] = random.uniform(-3.14,0)
        sample[link+3,0] = random.uniform(-5,5)
        sample[link+6,0] = round(random.uniform(-2,2), 1)
    return sample

@ray.remote
def generate_y(arm_teacher, sample):
    sample_state = sample[0:6,:]
    sample_action = sample[6:,:]
    next_state = arm_teacher.dynamics.advance(sample_state, sample_action)
    return next_state
"""    
def generate_samples(X, Y, arm_teacher, n):
    X_samples = []
    Y_samples = []
    
    for i in range(0,n):
        print(i)

        nlinks = arm_teacher.dynamics.get_num_links()
        
        sample = np.zeros((arm_teacher.dynamics.get_state_dim() + arm_teacher.dynamics.get_action_dim(),1))
        
        for link in range(0, nlinks):
            #sample[link,0] = random.uniform(-3.14,0) # for 1link
            #sample[link,0] = random.uniform(-3,1.3) # for 2link
            sample[link,0] = random.uniform(-3.3,1.3) # for 3link
            sample[link+nlinks,0] = random.uniform(-6,5) # for 3link
            #sample[link+nlinks*2,0] = round(random.uniform(-2,2), 2) # for 1link
            #sample[link+nlinks*2,0] = round(random.uniform(-1,1), 2) # for 2link
            sample[link+nlinks*2,0] = round(random.uniform(-2,1.4), 2) # for 3link
        X_samples.append(sample)

            
        sample_state = sample[0:nlinks*2,:]
        sample_action = sample[nlinks*2:,:]
        next_state = arm_teacher.dynamics.advance(sample_state, sample_action)
        Y_samples.append(next_state)

    X = np.hstack((X_samples))
    Y = np.hstack((Y_samples))
    return X, Y


if __name__ == '__main__':
    args = get_args()

    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=args.num_links,
        link_mass=args.link_mass,
        link_length=args.link_length,
        joint_viscous_friction=args.friction,
        dt=args.time_step
    )
    arm_teacher = Robot(dynamics_teacher)

    # ---
    # You code goes here. Replace the X, and Y by your collected data
    # Control the arm to collect a dataset for training the forward dynamics.

    print("Starting data collection")
    
    X = np.zeros((arm_teacher.dynamics.get_state_dim() + arm_teacher.dynamics.get_action_dim(), 0))
    Y = np.zeros((arm_teacher.dynamics.get_state_dim(), 0))

    X, Y  = generate_samples(X,Y,arm_teacher,sample_number)

    #print("Generating x...")
    #X_futures = [generate_x.remote(arm_teacher) for _ in range(sample_number)]
    #X_list = ray.get(X_futures)

    #print("Generating y...")
    #Y_futures = [generate_y.remote(arm_teacher, x) for x in X_list]
    #Y_list = ray.get(Y_futures)

    #X = np.hstack(X_list)
    #Y = np.hstack(Y_list)

    print("DONE")
    
    print(X.shape)
    print(Y.shape)

    # ---

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(os.path.join(args.save_dir, 'X.npy'), X)
    np.save(os.path.join(args.save_dir, 'Y.npy'), Y)
