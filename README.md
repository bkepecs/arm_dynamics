# arm_dynamics
This is a deep-learning forward dynamics system for robotic arms. I've trained a DL model to learn forward dynamics for a robotic arm of 1, 2, and 3 links and exert torques on joints to reach certain points.

To run the program, create a conda virtual environment with the dependencies in spec-file.txt and in the environment run: 

python mpc.py --gui --num_links 2 --model_path models/2link.pth

The "2" and "2link.pth" can be replaced with the equivalents for 1 and 3 links.

