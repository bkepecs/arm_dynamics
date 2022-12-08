from arm_dynamics_base import ArmDynamicsBase
import numpy as np
import torch
from models import build_model


class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, time_step, device):
        self.model = build_model(num_links, time_step)
        self.model.load_state_dict(torch.load(model_path))
        self.model_loaded = True
        self.device = device

    def dynamics_step(self, state, action, dt):
        if self.model_loaded:
            self.model.eval()
            X = np.concatenate((state,action), axis=0)
            X = torch.from_numpy(np.transpose(X)).float()
            new_state = self.model(X).detach().numpy()
            new_state = np.transpose(new_state)
            return new_state
        else:
            return state

