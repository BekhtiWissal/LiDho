import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

        self.homeostatic_plasticity = bool(getattr(args, "homeostatic_plasticity", False))
        if self.homeostatic_plasticity:
            # Homeostatic state
            self.register_buffer("h_running_mean", torch.zeros(args.hidden_dim))
            self.register_buffer("h_gain", torch.ones(args.hidden_dim))
            self.register_buffer("forward_steps", torch.zeros(1))

            # Hyperparameters
            self.homeo_start_t = int(getattr(args, "homeo_start_t", 50000))
            self.homeo_lr = float(getattr(args, "homeo_lr", 1e-4))
            self.homeo_beta = float(getattr(args, "homeo_beta", 1e-3))
            self.h_target = float(getattr(args, "homeo_target", 0.7))

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.args.hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))

        if self.homeostatic_plasticity:
            h = self._homeostatic_scaling(h)

        q = self.fc2(h)
        return q, h

    def _homeostatic_scaling(self, h):
        # ---- HOMEOSTATIC ACTIVATION SCALING ----
        if self.training:
            self.forward_steps += 1

            if self.forward_steps.item() > self.homeo_start_t:
                with torch.no_grad():
                    batch_mean = h.abs().mean(dim=0)

                    self.h_running_mean = (
                        (1 - self.homeo_beta) * self.h_running_mean
                        + self.homeo_beta * batch_mean
                    )

                    delta = self.h_target - self.h_running_mean
                    self.h_gain = (
                        self.h_gain * torch.exp(self.homeo_lr * delta)
                    ).clamp(0.5, 2.0).detach()

        return h * self.h_gain
