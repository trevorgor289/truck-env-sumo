"""
Lateral (lane-change) controller for the ego vehicle.

Issues a SUMO lane-change command and then steps the simulation for the time
it takes to physically cross one lane at a fixed lateral speed, tracking the
energy spent on the maneuver.
"""

import math
import traci

from . import energy_calc

EGO_LATERAL_SPEED = 0.8  # m/s -- TRUCK-SPECIFIC (this truck's assumed lateral maneuver speed)


class LateralController:
    def __init__(self, ego_id, lane_width, sim_step_length):
        self.ego_id = ego_id
        self.lane_width = lane_width
        self.sim_step_length = sim_step_length
        self.energy_consumed = 0.0

    def change_to_left_lane(self):
        traci.vehicle.changeLaneRelative(self.ego_id, -1, 1e15)
        self._run()

    def change_to_right_lane(self):
        traci.vehicle.changeLaneRelative(self.ego_id, 1, 1e15)
        self._run()

    def _run(self):
        duration = self.lane_width / EGO_LATERAL_SPEED
        steps = math.ceil(duration / self.sim_step_length)
        for _ in range(steps):
            traci.simulationStep()
        self.energy_consumed = energy_calc.calculate_energy_consumed(
            0, EGO_LATERAL_SPEED, traci.vehicle.getSlope(self.ego_id), duration)
