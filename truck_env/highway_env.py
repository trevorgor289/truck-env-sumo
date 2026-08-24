"""
Gymnasium environment for tactical decision-making of an autonomous truck on
a straight highway in SUMO.

Action space and controller split are modeled on the reference repo's
`new_architecture`: the RL agent picks a semantic action (target time-gap,
speed nudge, lane change, or maintain); a longitudinal (IDM-based ACC) and a
lateral controller translate that into TraCI vehicle commands over several
SUMO physics substeps. Rewritten against SUMO 1.27.1 and the Gymnasium
reset/step signatures (terminated/truncated split, seeded RNG) rather than
copied from the reference repo, which targets SUMO 1.15.0 and the old `gym`
package.
"""

import copy
import os
import warnings

import numpy as np
from gymnasium import Env, spaces

import traci
from sumolib import checkBinary

from .road import Road
from .longitudinal_controller import LongitudinalController
from .lateral_controller import LateralController

# Semantic action ids
SET_TARGET_VEH_SHORT_GAP = 0
SET_TARGET_VEH_MEDIUM_GAP = 1
SET_TARGET_VEH_LONG_GAP = 2
INCREASE_DESIRED_SPEED = 3
DECREASE_DESIRED_SPEED = 4
CHANGE_LANE_LEFT = 5
CHANGE_LANE_RIGHT = 6
MAINTAIN_SPEED_AND_GAP = 7
ACTION_COUNT = 8

# TraCI vehicle subscription constants (see traci.constants)
POSITION = 66
LONG_SPEED = 64
LAT_SPEED = 50
LONG_ACC = 114
LANE_INDEX = 82
SIGNALS = 91
ROAD_ID = 80

NB_EGO_STATES = 6
NB_STATES_PER_VEHICLE = 7


class TruckHighwayEnv(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, sim_params, road_params, sumo_home=None, use_gui=True,
                 road_path=None, name_suffix=""):
        super().__init__()

        self.sumo_home = sumo_home or os.environ["SUMO_HOME"]
        self.sim_params = sim_params
        self.use_gui = use_gui

        self.sim_step_length = sim_params["sim_step_length"]
        self.long_control_duration = sim_params["long_control_duration"]
        self.max_steps = sim_params["max_steps"]
        self.init_steps = sim_params["init_steps"]
        self.nb_vehicles = sim_params["nb_vehicles"]
        self.safety_check = sim_params["safety_check"]

        road_path = road_path or os.path.join(os.getcwd(), "road")
        self.road = Road(road_params, road_path, self.sumo_home, name_suffix=name_suffix)
        self.road.create_road()

        self.nb_lanes = road_params["nb_lanes"]
        self.lane_width = road_params["lane_width"]
        self.speed_range = road_params["speed_range"]
        self.max_allowed_ego_speed = min(road_params["vehicles"][0]["maxSpeed"], road_params["max_road_speed"])
        self.min_allowed_ego_speed = road_params["min_road_speed"]

        self.sensor_range = sim_params["sensor_range"]
        self.sensor_nb_vehicles = sim_params["sensor_nb_vehicles"]
        self.target_gap = {
            SET_TARGET_VEH_SHORT_GAP: sim_params["target_veh_short_gap"],
            SET_TARGET_VEH_MEDIUM_GAP: sim_params["target_veh_medium_gap"],
            SET_TARGET_VEH_LONG_GAP: sim_params["target_veh_long_gap"],
        }
        self.cruise_acceleration = sim_params["cruise_acceleration"]
        self.cruise_deceleration = sim_params["cruise_deceleration"]

        self.collision_penalty = sim_params["collision_penalty"]
        self.near_collision_penalty = sim_params["near_collision_penalty"]
        self.outside_road_penalty = sim_params["outside_road_penalty"]
        self.lane_change_penalty = sim_params["lane_change_penalty"]
        self.completion_reward = sim_params["completion_reward"]

        state_size = NB_EGO_STATES + self.sensor_nb_vehicles * NB_STATES_PER_VEHICLE
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float64)
        self.action_space = spaces.Discrete(ACTION_COUNT)

        self.ego_id = "veh" + str(0).zfill(int(np.ceil(np.log10(self.nb_vehicles))))
        self.positions = np.zeros([self.nb_vehicles, 2])
        self.speeds = np.zeros([self.nb_vehicles, 2])
        self.lanes = np.zeros([self.nb_vehicles])
        self.signals = np.zeros([self.nb_vehicles])
        self.edge_name = np.empty([self.nb_vehicles], dtype="object")
        self.vehicles = []

        self.long_controller = LongitudinalController(
            self.ego_id, self.max_allowed_ego_speed, self.min_allowed_ego_speed,
            self.target_gap[SET_TARGET_VEH_MEDIUM_GAP], self.sim_step_length, self.long_control_duration)
        self.lat_controller = LateralController(self.ego_id, self.lane_width, self.sim_step_length)

        # Episode-level eval metrics
        self.nb_collisions = 0
        self.nb_near_collisions = 0
        self.nb_outside_road = 0
        self.nb_max_step = 0
        self.nb_reached_exit = 0
        self.step_ = 0
        self.ego_energy_cost_kwh = 0.0

        sumo_binary = checkBinary("sumo-gui" if use_gui else "sumo")
        start_cmd = [sumo_binary, "-c", self.road.sumocfg_path, "--start",
                     "--step-length", str(self.sim_step_length)]
        if sim_params["remove_sumo_warnings"]:
            start_cmd.append("--no-warnings")
        traci.start(start_cmd)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        for veh in traci.vehicle.getIDList():
            traci.vehicle.unsubscribe(veh)
            traci.vehicle.remove(veh)
        traci.simulationStep()

        for i in range(self.nb_vehicles):
            veh_id = "veh" + str(i).zfill(int(np.ceil(np.log10(self.nb_vehicles))))
            lane = i % self.nb_lanes
            traci.vehicle.add(veh_id, "route0", typeID="truck" if i == 0 else "car",
                               departLane=lane, departPos="base",
                               departSpeed=self.max_allowed_ego_speed)
            if (i + 1) % self.nb_lanes == 0:
                traci.simulationStep()
                for veh in traci.vehicle.getIDList():
                    traci.vehicle.moveTo(veh, traci.vehicle.getLaneID(veh),
                                          traci.vehicle.getLanePosition(veh) + 50.0)
        traci.simulationStep()
        assert len(traci.vehicle.getIDList()) == self.nb_vehicles, \
            "Not all vehicles could be inserted -- check road length vs nb_vehicles"
        self.vehicles = traci.vehicle.getIDList()

        start_edge = self.road.road_params["edges"][1]
        start_lane_length = traci.lane.getLength(start_edge + "_0")
        x_pos = self.np_random.uniform(0.0, start_lane_length, self.nb_vehicles)
        x_pos[0] = start_lane_length / 2
        lanes = self.np_random.integers(0, self.nb_lanes, self.nb_vehicles)

        init_speed = np.zeros(self.nb_vehicles)
        leaders = x_pos > x_pos[0]
        followers = x_pos < x_pos[0]
        init_speed[0] = self.max_allowed_ego_speed
        if np.any(leaders):
            init_speed[leaders] = self.np_random.uniform(
                self.speed_range[0], init_speed[0], np.sum(leaders))
        if np.any(followers):
            init_speed[followers] = self.np_random.uniform(
                init_speed[0], self.speed_range[1], np.sum(followers))

        for i, veh in enumerate(self.vehicles):
            traci.vehicle.moveTo(veh, f"{start_edge}_{lanes[i]}", x_pos[i])
            traci.vehicle.setSpeed(veh, init_speed[i])
            if i != 0:
                traci.vehicle.setSpeedMode(veh, 1)
                traci.vehicle.setMaxSpeed(veh, init_speed[i])

        if not self.safety_check:
            traci.vehicle.setSpeedMode(self.ego_id, 0)
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)

        traci.vehicle.setAcceleration(self.ego_id, 0, self.init_steps * self.sim_step_length)

        for veh in self.vehicles:
            traci.vehicle.subscribe(veh, [POSITION, LONG_SPEED, LAT_SPEED, LONG_ACC, LANE_INDEX, SIGNALS, ROAD_ID])

        for _ in range(self.init_steps):
            traci.simulationStep()

        for veh in self.vehicles[1:]:
            traci.vehicle.setSpeed(veh, -1)  # give speed control back to SUMO

        if self.use_gui:
            traci.gui.trackVehicle("View #0", self.ego_id)

        self.step_ = 0
        self.ego_energy_cost_kwh = self.long_controller.initial_kinetic_energy(self.max_allowed_ego_speed) * 0.5
        self.long_controller.desired_speed = self.max_allowed_ego_speed
        self.long_controller.desired_time_gap = self.target_gap[SET_TARGET_VEH_MEDIUM_GAP]

        self._update_vehicle_state()
        observation = self._sensor_model()
        info = {}
        return observation, info

    def step(self, action):
        self.step_ += 1
        outside_road = False
        terminated = False
        truncated = False
        events = []

        if action == CHANGE_LANE_LEFT:
            if self.lanes[0] == 0:
                terminated = True
                outside_road = True
                events.append("outside_road")
            else:
                self.lat_controller.change_to_left_lane()
        elif action == CHANGE_LANE_RIGHT:
            if self.lanes[0] == self.nb_lanes - 1:
                terminated = True
                outside_road = True
                events.append("outside_road")
            else:
                self.lat_controller.change_to_right_lane()
        elif action in self.target_gap:
            self.long_controller.change_desired_time_gap(self.target_gap[action])
        elif action == INCREASE_DESIRED_SPEED:
            self.long_controller.change_desired_speed(self.cruise_acceleration)
        elif action == DECREASE_DESIRED_SPEED:
            self.long_controller.change_desired_speed(self.cruise_deceleration)
        elif action == MAINTAIN_SPEED_AND_GAP:
            self.long_controller.maintain_speed_and_gap()
        else:
            raise ValueError(f"Undefined action: {action}")

        self._update_vehicle_state()

        ego_collision, ego_near_collision, collision_info = self._check_collisions()
        if collision_info:
            events.extend(collision_info)
        if ego_collision:
            terminated = True

        if self.edge_name[0] == "exit":
            terminated = True
            events.append("reached_exit")
        elif self.step_ >= self.max_steps:
            truncated = True
            events.append("max_steps")

        if outside_road:
            self.nb_outside_road += 1
        elif ego_collision:
            self.nb_collisions += 1
        elif ego_near_collision:
            self.nb_near_collisions += 1
        elif truncated:
            self.nb_max_step += 1
        elif self.edge_name[0] == "exit":
            self.nb_reached_exit += 1

        state_done = terminated or truncated
        state = copy.deepcopy([self.positions, self.speeds, self.lanes, self.signals, self.edge_name, state_done])
        observation = self._sensor_model()
        reward = self._reward_model(state, action, ego_collision, ego_near_collision, outside_road)

        info = {"events": events}
        return observation, reward, terminated, truncated, info

    def _update_vehicle_state(self):
        nb_digits = int(np.floor(np.log10(self.nb_vehicles))) + 1
        for veh in self.vehicles:
            i = int(veh[-nb_digits:])
            out = traci.vehicle.getSubscriptionResults(veh)
            if not out:
                continue
            self.positions[i, :] = np.array(out[POSITION]) + \
                np.array([0, self.lane_width * self.nb_lanes - self.lane_width / 2])
            self.speeds[i, 0] = out[LONG_SPEED]
            self.speeds[i, 1] = out[LAT_SPEED]
            self.lanes[i] = out[LANE_INDEX]
            self.signals[i] = out[SIGNALS]
            self.edge_name[i] = out[ROAD_ID]

    def _check_collisions(self):
        ego_collision = False
        ego_near_collision = False
        info = []
        if traci.simulation.getCollidingVehiclesNumber() == 0:
            return ego_collision, ego_near_collision, info

        colliding_ids = traci.simulation.getCollidingVehiclesIDList()
        if self.ego_id not in colliding_ids:
            return ego_collision, ego_near_collision, info

        other_id = colliding_ids[1] if colliding_ids[0] == self.ego_id else colliding_ids[0]
        ego_pos = traci.vehicle.getPosition(self.ego_id)[0]
        other_pos = traci.vehicle.getPosition(other_id)[0]
        long_dist = other_pos - ego_pos if other_id == colliding_ids[1] else ego_pos - other_pos

        # A "collision" that's really just a tight-but-legal gap in front is a near-collision, not a crash.
        if long_dist > 0:
            other_len = traci.vehicle.getLength(other_id)
            if long_dist - other_len > 0:
                ego_near_collision = True
                info.append("near_collision")
                return ego_collision, ego_near_collision, info

        if self.step_ == 0:
            warnings.warn("Collision during reset phase; this should not happen.")
        else:
            ego_collision = True
            info.append("collision")
        return ego_collision, ego_near_collision, info

    def _reward_model(self, state, action, ego_collision, ego_near_collision, outside_road):
        reward = state[1][0, 0] / self.max_allowed_ego_speed
        if action in (CHANGE_LANE_LEFT, CHANGE_LANE_RIGHT):
            reward -= self.lane_change_penalty

        if outside_road:
            reward -= self.outside_road_penalty
        elif ego_near_collision:
            reward -= self.near_collision_penalty
        elif ego_collision:
            reward -= self.collision_penalty
        elif state[4][0] == "exit":
            reward += self.completion_reward / self.step_

        energy = (self.lat_controller.energy_consumed
                  if action in (CHANGE_LANE_LEFT, CHANGE_LANE_RIGHT)
                  else self.long_controller.energy_consumed)
        self.ego_energy_cost_kwh += energy * 0.5  # placeholder $/kWh weighting, see TCOP variant later
        return reward

    def _sensor_model(self):
        rel_long = self.positions[1:, 0] - self.positions[0, 0]
        in_range = np.abs(rel_long) <= self.sensor_range
        if np.sum(in_range) > self.sensor_nb_vehicles:
            warnings.warn("More vehicles within sensor range than the sensor can represent")

        obs = np.zeros(NB_EGO_STATES + NB_STATES_PER_VEHICLE * self.sensor_nb_vehicles)
        obs[0] = self.speeds[0, 0]
        obs[1] = np.sign(self.speeds[0, 1])
        obs[2] = self.lanes[0]
        obs[3] = 1 if int(self.signals[0]) & 2 else 0
        obs[4] = 1 if int(self.signals[0]) & 1 else 0

        leader = traci.vehicle.getLeader(self.ego_id)
        obs[5] = leader[1] if leader is not None else 1e6

        idx = 0
        for i, within_range in enumerate(in_range):
            if not within_range or idx >= self.sensor_nb_vehicles:
                continue
            base = NB_EGO_STATES + idx * NB_STATES_PER_VEHICLE
            obs[base + 0] = self.positions[i + 1, 0] - self.positions[0, 0]
            obs[base + 1] = self.positions[i + 1, 1] - self.positions[0, 1]
            obs[base + 2] = self.speeds[i + 1, 0] - self.speeds[0, 0]
            obs[base + 3] = np.sign(self.speeds[i + 1, 1])
            obs[base + 4] = self.lanes[i + 1]
            obs[base + 5] = 1 if int(self.signals[i + 1]) & 2 else 0
            obs[base + 6] = 1 if int(self.signals[i + 1]) & 1 else 0
            idx += 1

        return obs

    def close(self):
        traci.close()
