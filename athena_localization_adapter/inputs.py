from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class GyroBias:
    alpha: float = 0.02
    still_time: float = 1.0
    max_rate: float = 0.01
    standstill_speed: float = 0.05

    bias: float = field(default=0.0, init=False)
    _count: int = field(default=0, init=False, repr=False)
    _still_since: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.still_time < 0.0 or self.max_rate <= 0.0 or self.standstill_speed < 0.0:
            raise ValueError(
                "still_time, standstill_speed must be >= 0 and max_rate > 0"
            )

    def wheel(self, stamp: float, speed: float) -> None:
        """Wheel speed [m/s] at time `stamp` [s]."""
        if abs(speed) < self.standstill_speed:
            if self._still_since is None:
                self._still_since = stamp
        else:
            self._still_since = None

    def standing(self, stamp: float) -> bool:
        """Whether the vehicle has been standing for at least `still_time`."""
        return (
            self._still_since is not None
            and stamp - self._still_since >= self.still_time
        )

    def correct(self, stamp: float, rate: float) -> float:
        """Yaw rate [rad/s] without bias; learns the bias while the vehicle is standing."""
        if self.standing(stamp) and abs(rate - self.bias) < self.max_rate:
            self._count += 1
            self.bias += max(self.alpha, 1.0 / self._count) * (rate - self.bias)
        return rate - self.bias


@dataclass
class Inputs:
    """Conversion of IMU and wheel data, variances for the EKF."""

    gyro_scale: float = math.pi / 180.0  # raw -> rad/s
    rate_variance: float = 1e-5  # yaw rate [(rad/s)^2]
    speed_variance: float = 4e-4  # v [(m/s)^2]
    lateral_variance: float = 1e-4  # lateral v = 0 [(m/s)^2]
    bias: GyroBias = field(default_factory=GyroBias)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "Inputs":
        """From the node parameters; `bias.*` for `GyroBias`."""
        bias = {
            key[len("bias.") :]: float(value)
            for key, value in params.items()
            if key.startswith("bias.")
        }
        known = (
            "gyro_scale",
            "rate_variance",
            "speed_variance",
            "lateral_variance",
        )
        return cls(
            **{key: float(params[key]) for key in known if key in params},
            bias=GyroBias(**bias)
        )

    def rates(self, stamp: float, raw: tuple) -> tuple:
        """Angular rates (x, y, z) [rad/s] from raw values; z without bias."""
        x, y, z = (self.gyro_scale * value for value in raw)
        return x, y, self.bias.correct(stamp, z)

    def speed(self, stamp: float, speed: float) -> float:
        """Wheel speed [m/s]; remembers whether the vehicle is standing."""
        self.bias.wheel(stamp, speed)
        return speed
