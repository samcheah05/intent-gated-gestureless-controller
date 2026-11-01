import math

class OneEuroFilter:
    """
    Biomechanical 1-Euro Filter (Casiez et al., 2012)
    Decouples low-velocity physiological tremor suppression (8-12 Hz)
    from high-velocity tracking latency.
    """
    def __init__(self, t0: float, x0: float, dx0: float = 0.0,
                 min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    @staticmethod
    def _alpha(rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, t: float, x: float) -> float:
        t_e = t - self.t_prev
        if t_e <= 1e-6:
            return self.x_prev
        rate = 1.0 / t_e

        # 1. Discrete velocity derivative & low-pass filter
        dx = (x - self.x_prev) * rate
        alpha_d = self._alpha(rate, self.d_cutoff)
        dx_star = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # 2. Adaptive cutoff frequency modulation
        fc_hat = self.min_cutoff + self.beta * abs(dx_star)

        # 3. Coordinate exponential smoothing
        alpha = self._alpha(rate, fc_hat)
        x_star = alpha * x + (1.0 - alpha) * self.x_prev

        self.t_prev = t
        self.x_prev = x_star
        self.dx_prev = dx_star
        return x_star


class Point2DOneEuroFilter:
    """Wrapper for simultaneous (X, Y) pointer stabilization."""
    def __init__(self, t0: float, x0: float, y0: float,
                 min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.fx = OneEuroFilter(t0, x0, 0.0, min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(t0, y0, 0.0, min_cutoff, beta, d_cutoff)

    def filter(self, t: float, x: float, y: float):
        return self.fx.filter(t, x), self.fy.filter(t, y)
