import math
import torch

from dataclasses import dataclass

@dataclass(frozen=True)
class ActuatorSpec:
    rotor_inertia: float
    gear_ratio: float

    @property
    def armature(self) -> float:
        return self.rotor_inertia * self.gear_ratio ** 2


class BezierCurve:
    def __init__(self, points=None, n_envs=1, dim=3, n_points=5):
        if points is None:
            self.initialize_random(n_envs, dim, n_points)
        else:
            self.setup(points)

    def initialize_random(self, n_envs, dim, n_points):
        points = []
        for _ in range(n_points):
            shape = (dim,) if n_envs == 1 else (n_envs, dim)
            b_point = torch.rand(shape)
            points.append(b_point)
        self.setup(points)

    def setup(self, points):
        if isinstance(points, (list, tuple)):
            self.return_numpy = not any(torch.is_tensor(p) for p in points)
        else:
            self.return_numpy = not torch.is_tensor(points)

        if isinstance(points, (list, tuple)):
            points = torch.stack([torch.as_tensor(p) for p in points], dim=0)
        else:
            points = torch.as_tensor(points)

        self.points = points
        self.order = points.shape[0] - 1
        self.binomial = torch.tensor(
            [math.comb(self.order, i) for i in range(self.order + 1)],
            device=points.device,
            dtype=points.dtype,
        )
        self.binomial_d = torch.tensor(
            [math.comb(self.order - 1, i) for i in range(self.order)],
            device=points.device,
            dtype=points.dtype,
        ) if self.order >= 1 else points.new_zeros(1)
        self.binomial_dd = torch.tensor(
            [math.comb(self.order - 2, i) for i in range(self.order - 1)],
            device=points.device,
            dtype=points.dtype,
        ) if self.order >= 2 else points.new_zeros(1)

    def sample(self, t):
        t = torch.as_tensor(t, device=self.points.device, dtype=self.points.dtype)
        scalar_t = t.ndim == 0
        if scalar_t:
            t = t.unsqueeze(0)

        n = self.order
        extra = (1,) * (self.points.ndim - 1)

        # Position
        i = torch.arange(n + 1, device=t.device, dtype=t.dtype)
        basis = self.binomial.view(1, -1) * torch.pow(1 - t.unsqueeze(-1), n - i) * torch.pow(t.unsqueeze(-1), i)
        samples = (basis.view(t.shape[0], n + 1, *extra) * self.points.unsqueeze(0)).sum(dim=1)

        # Velocity (1st derivative): n-th order Bezier derivative is order n-1
        diff_pts = self.points[1:] - self.points[:-1]  # (n, ...)
        i_d = torch.arange(n, device=t.device, dtype=t.dtype)
        basis_d = self.binomial_d.view(1, -1) * torch.pow(1 - t.unsqueeze(-1), n - 1 - i_d) * torch.pow(t.unsqueeze(-1), i_d)
        vels = n * (basis_d.view(t.shape[0], n, *extra) * diff_pts.unsqueeze(0)).sum(dim=1)

        # Acceleration (2nd derivative): order n-2
        diff2_pts = diff_pts[1:] - diff_pts[:-1]  # (n-1, ...)
        i_dd = torch.arange(n - 1, device=t.device, dtype=t.dtype)
        basis_dd = self.binomial_dd.view(1, -1) * torch.pow(1 - t.unsqueeze(-1), n - 2 - i_dd) * torch.pow(t.unsqueeze(-1), i_dd)
        accs = n * (n - 1) * (basis_dd.view(t.shape[0], n - 1, *extra) * diff2_pts.unsqueeze(0)).sum(dim=1)

        if scalar_t:
            samples, vels, accs = samples[0], vels[0], accs[0]
        if self.return_numpy:
            to_np = lambda x: x.detach().cpu().numpy()
            return to_np(samples), to_np(vels), to_np(accs)
        return samples, vels, accs

    __call__ = sample
