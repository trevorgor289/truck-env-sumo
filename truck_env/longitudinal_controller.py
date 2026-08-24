"""
Longitudinal (speed/gap) controller for the ego vehicle.

Translates the RL agent's semantic action (desired speed change, desired
time-gap) into per-substep accelerations via the Intelligent Driver Model
(IDM), then applies them through TraCI. One call from the env corresponds to
`long_control_duration / sim_step_length` SUMO physics steps.
"""

import numpy as np
import traci

from . import energy_calc

IDM_EXPONENT = 4


class LongitudinalController:
    def __init__(self, ego_id, max_allowed_speed, min_allowed_speed,
                 desired_time_gap, sim_step_length, long_control_duration):
        self.ego_id = ego_id
        self.max_allowed_speed = max_allowed_speed
        self.min_allowed_speed = min_allowed_speed
        self.sim_step_length = sim_step_length
        self.control_step_count = round(long_control_duration / sim_step_length)
        self.desired_speed = max_allowed_speed
        self.desired_time_gap = desired_time_gap
        self.energy_consumed = 0.0

    def change_desired_speed(self, delta):
        self.desired_speed = float(np.clip(self.desired_speed + delta,
                                            self.min_allowed_speed, self.max_allowed_speed))
        self._run()

    def change_desired_time_gap(self, time_gap):
        self.desired_time_gap = time_gap
        self._run()

    def maintain_speed_and_gap(self):
        self._run()

    def initial_kinetic_energy(self, speed):
        return energy_calc.calculate_init_kinetic_energy(speed)

    def _idm_accel(self):
        ego_speed = traci.vehicle.getSpeed(self.ego_id)
        ego_x = traci.vehicle.getPosition(self.ego_id)[0]
        max_accel = traci.vehicle.getAccel(self.ego_id)
        max_decel = traci.vehicle.getDecel(self.ego_id)

        leader = traci.vehicle.getLeader(self.ego_id)
        free_road_term = 1 - (ego_speed / self.desired_speed) ** IDM_EXPONENT

        if leader is None:
            accel = max_accel * free_road_term
        else:
            leader_id, gap = leader
            leader_speed = traci.vehicle.getSpeed(leader_id)
            min_gap = traci.vehicle.getMinGap(self.ego_id)
            delta_v = ego_speed - leader_speed
            s = max(gap, 0.1)  # avoid div by zero if SUMO reports overlap
            s_star = min_gap + max(0.0, ego_speed * self.desired_time_gap +
                                    (ego_speed * delta_v) / (2 * np.sqrt(max_accel * max_decel)))
            accel = max_accel * (free_road_term - (s_star / s) ** 2)

        return float(np.clip(accel, -max_decel, max_accel))

    def _run(self):
        self.energy_consumed = 0.0
        for _ in range(self.control_step_count):
            current_speed = traci.vehicle.getSpeed(self.ego_id)
            accel = self._idm_accel()
            new_speed = max(0.0, current_speed + accel * self.sim_step_length)
            self.energy_consumed += energy_calc.calculate_energy_consumed(
                accel, current_speed, traci.vehicle.getSlope(self.ego_id), self.sim_step_length)
            traci.vehicle.setSpeed(self.ego_id, new_speed)
            traci.simulationStep()
