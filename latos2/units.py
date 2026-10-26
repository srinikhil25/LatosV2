"""Unit handling for logger channels: the header unit is the only authority.

The workbook's bracketed unit decides the conversion; nothing is inferred from
the size of the numbers. Verified for the ionic-liquid campaign on 2026-10-26:
the voltage channel is headed ``[V]``, its readings step by 5e-6 in that unit
(5 µV, the finest resolution of the logger's ±100 mV range), and the
researcher's own Origin fits (slope 0.00283 per K on cycle 1 heating) agree
only with volts. An axis reading 0-40 on those plots is millivolts.
"""

from __future__ import annotations

from decimal import Decimal
from math import gcd

import numpy as np

__all__ = [
    "UnitError",
    "check_temperature_unit",
    "quantization_step",
    "voltage_factor_to_mv",
]

# Multiply a reading in the header unit by this to get millivolts.
_VOLTAGE_TO_MV = {
    "V": 1000.0,
    "mV": 1.0,
    "uV": 1e-3,
    "µV": 1e-3,  # micro sign U+00B5
    "μV": 1e-3,  # Greek mu U+03BC
}

# A difference of two readings in any of these is a difference in kelvin.
_TEMPERATURE_UNITS = {"°C", "℃", "C", "degC", "K"}

# Readings are compared on a grid far finer than any logger step, so binary
# float noise (1.6000000000000001E-3) does not hide the true step.
_STEP_DECIMALS = 12


class UnitError(ValueError):
    """A channel's unit is not one this analysis accepts."""


def voltage_factor_to_mv(unit: str) -> float:
    """The factor that turns a reading in `unit` into millivolts."""
    try:
        return _VOLTAGE_TO_MV[unit.strip()]
    except KeyError:
        accepted = ", ".join(sorted(_VOLTAGE_TO_MV))
        raise UnitError(f"voltage unit {unit!r} is not accepted; expected one of {accepted}") from None


def check_temperature_unit(unit: str) -> str:
    """Return `unit` if a difference of two readings in it is in kelvin."""
    if unit.strip() not in _TEMPERATURE_UNITS:
        raise UnitError(
            f"temperature unit {unit!r} is not accepted; expected °C or K, so that "
            "T_hot − T_cold is a difference in kelvin"
        )
    return unit.strip()


def quantization_step(values: np.ndarray) -> float:
    """The smallest step the readings move in, in their own unit.

    The greatest common divisor of every reading on a 1e-12 grid. A logger
    writes multiples of its resolution, so this recovers the resolution, which
    is a check on the unit: 5e-6 in a ``[V]`` column is 5 µV.
    """
    scale = 10**_STEP_DECIMALS
    step = 0
    for value in np.asarray(values, dtype=np.float64):
        step = gcd(step, abs(int(round(Decimal(repr(float(value))) * scale))))
    return step / scale
