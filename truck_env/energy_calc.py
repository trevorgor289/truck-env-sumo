"""
Longitudinal energy-consumption model for the ego vehicle.

TRUCK-SPECIFIC: every constant below (mass, drag coefficient, frontal area,
rolling resistance) describes this particular truck and must be replaced when
adapting to a different vehicle (plan Phase 5).
"""

import math

MASS_KG = 40000          # ego vehicle mass
DRAG_COEFFICIENT = 0.36  # Cd
FRONTAL_AREA_M2 = 10.0   # Af
AIR_DENSITY = 1.225      # kg/m^3
ROLLING_RESISTANCE = 0.005  # Cr
GRAVITY = 9.81           # m/s^2


def calculate_energy_consumed(accel, speed, slope_pct, duration_s):
    """Energy consumed (kWh) over `duration_s` seconds at constant `accel`/`speed`/`slope`."""
    force = (
        (MASS_KG * accel)
        + (0.5 * DRAG_COEFFICIENT * FRONTAL_AREA_M2 * AIR_DENSITY * speed * speed)
        + (GRAVITY * ROLLING_RESISTANCE * MASS_KG)
        + (MASS_KG * GRAVITY * math.sin(math.atan(slope_pct / 100)))
    )
    power_w = force * speed
    return (power_w / 1000) * (duration_s / 3600)  # kWh


def calculate_init_kinetic_energy(speed):
    return (0.5 * MASS_KG * speed * speed) / (1000 * 3600)  # kWh
