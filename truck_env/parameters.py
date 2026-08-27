"""
Parameters for the truck highway environment.

Modeled on deepthi-pathare/Autonomous-truck-sumo-gym-env's `new_architecture`
(semantic actions + low-level longitudinal/lateral controllers), rewritten
against SUMO 1.27.1's TraCI API and the Gymnasium interface instead of the
reference repo's pinned SUMO 1.15.0 / old `gym` package.

Values tagged TRUCK-SPECIFIC are hardcoded to this particular vehicle and are
exactly what should change when adapting to a different scenario (plan Phase 5).
"""

import numpy as np

sim_params = {}
sim_params["sim_step_length"] = 0.1        # SUMO physics step, seconds
sim_params["long_control_duration"] = 1.0  # seconds of physics per longitudinal RL action
sim_params["max_steps"] = 500              # max RL steps per episode
sim_params["init_steps"] = int(4 * (sim_params["long_control_duration"] / sim_params["sim_step_length"]))
sim_params["nb_vehicles"] = 25              # was 15 -- denser traffic, tighter gaps
sim_params["remove_sumo_warnings"] = True
sim_params["safety_check"] = False         # if True, SUMO overrides "unsafe" ego decisions
sim_params["sensor_range"] = 200.0
sim_params["sensor_nb_vehicles"] = 25       # kept equal to nb_vehicles so the observation can still represent all of them in range

# Longitudinal controller (IDM-based ACC) targets, set by RL actions
sim_params["target_veh_short_gap"] = 1.0   # seconds
sim_params["target_veh_medium_gap"] = 2.0
sim_params["target_veh_long_gap"] = 3.0
sim_params["cruise_acceleration"] = 1.0    # m/s^2, applied to desired speed on INCREASE_DESIRED_SPEED
sim_params["cruise_deceleration"] = -1.0

# Reward
sim_params["collision_penalty"] = 10.0
sim_params["near_collision_penalty"] = 10.0
sim_params["outside_road_penalty"] = 10.0
sim_params["lane_change_penalty"] = 1.0
sim_params["completion_reward"] = 100.0

# --- Vehicle types ---
vehicles = []

# Vehicle 0 = ego truck. TRUCK-SPECIFIC: every field below.
vehicles.append({
    "id": "truck",
    "vClass": "trailer",
    "length": 16.0,
    "width": 2.55,
    "maxSpeed": 25.0,
    "speedFactor": 1.0,
    "speedDev": 0,
    "carFollowModel": "Krauss",
    "minGap": 2.5,
    "accel": 1.1,
    "decel": 4.0,
    "emergencyDecel": 9.0,
    "sigma": 0.0,
    "tau": 1.0,
    "color": "1,0,0",
    "laneChangeModel": "LC2013",
    "lcStrategic": 0,
    "lcCooperative": 0,
    "lcSpeedGain": 1.0,
    "lcKeepRight": 0,
    "lcOvertakeRight": 0,
    "lcOpposite": 1.0,
    "lcLookaheadLeft": 2.0,
    "lcSpeedGainRight": 1.0,
    "lcAssertive": 1.0,
    "lcMaxSpeedLatFactor": 1.0,
    "lcSigma": 0.0,
})

# Vehicle 1 = surrounding traffic (passenger cars). Not truck-specific.
vehicles.append({
    "id": "car",
    "vClass": "passenger",
    "length": 4.8,
    "width": 1.8,
    "maxSpeed": 100.0,  # overwritten per-vehicle at reset
    "speedFactor": 1.0,
    "speedDev": 0,
    "carFollowModel": "Krauss",
    "minGap": 2.5,
    "accel": 2.6,
    "decel": 4.5,
    "emergencyDecel": 9.0,
    "sigma": 0.0,
    "tau": 1.0,
    "laneChangeModel": "LC2013",
    "lcStrategic": 0,
    "lcCooperative": 0,
    "lcSpeedGain": 1.0,
    "lcKeepRight": 0,
    "lcOvertakeRight": 0,
    "lcOpposite": 1.0,
    "lcLookaheadLeft": 2.0,
    "lcSpeedGainRight": 1.0,
    "lcAssertive": 1.0,
    "lcMaxSpeedLatFactor": 1.0,
    "lcSigma": 0.0,
})

road_params = {}
road_params["name"] = "highway"
road_params["nb_lanes"] = 3
road_params["lane_width"] = 3.2
road_params["max_road_speed"] = 100.0   # set high; actual cap comes from vType maxSpeed
road_params["min_road_speed"] = 1.0
road_params["lane_change_duration"] = 4
road_params["speed_range"] = np.array([10, 40])   # surrounding traffic speed range, m/s (was [15,35] -- wider/less predictable)
road_params["overtake_right"] = "true"
road_params["nodes"] = np.array([[0.0, 0.0], [400.0, 0.0], [1000.0, 0.0], [3000.0, 0.0], [5000.0, 0.0]])
road_params["edges"] = ["add", "start", "highway", "exit"]
road_params["vehicles"] = vehicles
road_params["collision_action"] = "warn"

# Terminal output
road_params["emergency_decel_warn_threshold"] = 10
road_params["no_display_step"] = "true"

# GUI view settings
road_params["view_position"] = np.array([750, 0])
road_params["zoom"] = 3500
road_params["view_delay"] = 200
road_params["info_pos"] = [0, 35]
