"""
Generates the SUMO network/route/config XML files for the highway scenario
and invokes `netconvert` to build the .net.xml.

Modeled on the reference repo's Road class; rewritten to call netconvert via
subprocess (not os.system) and to point at the locally extracted SUMO 1.27.1
toolchain rather than a system-wide install.
"""

import os
import subprocess


class Road:
    def __init__(self, road_params, road_path, sumo_home, name_suffix=""):
        self.road_params = road_params
        self.road_path = road_path
        self.sumo_home = sumo_home
        self.name = road_params["name"] + (("_" + name_suffix) if name_suffix else "")

        os.makedirs(self.road_path, exist_ok=True)

    def _path(self, ext):
        return os.path.join(self.road_path, self.name + ext)

    def _write_nodes(self):
        with open(self._path(".nod.xml"), "w") as f:
            f.write("<nodes>\n")
            for idx, (x, y) in enumerate(self.road_params["nodes"]):
                f.write(f'   <node id="{idx}" x="{x}" y="{y}" />\n')
            f.write("</nodes>\n")

    def _write_edges(self):
        with open(self._path(".edg.xml"), "w") as f:
            f.write("<edges>\n")
            for idx, edge in enumerate(self.road_params["edges"]):
                f.write(
                    f'   <edge from="{idx}" id="{edge}" to="{idx + 1}" '
                    f'numLanes="{self.road_params["nb_lanes"]}" '
                    f'width="{self.road_params["lane_width"]}" '
                    f'speed="{self.road_params["max_road_speed"]}" />\n'
                )
            f.write("</edges>\n")

    def _write_routes(self):
        with open(self._path(".rou.xml"), "w") as f:
            f.write("<routes>\n")
            for vehicle in self.road_params["vehicles"]:
                attrs = " ".join(f'{k}="{v}"' for k, v in vehicle.items())
                f.write(f"   <vType {attrs} />\n")
            edges_string = " ".join(self.road_params["edges"])
            f.write(f'   <route id="route0" edges="{edges_string}"/>\n')
            f.write("</routes>\n")

    def _write_config(self):
        with open(self._path(".sumocfg"), "w") as f:
            f.write("<configuration>\n")
            f.write("   <input>\n")
            f.write(f'      <net-file value="{self.name}.net.xml"/>\n')
            f.write(f'      <route-files value="{self.name}.rou.xml"/>\n')
            f.write(f'      <gui-settings-file value="{self.name}.settings.xml"/>\n')
            f.write("   </input>\n")
            f.write("   <time>\n")
            f.write('      <begin value="0"/>\n')
            f.write('      <end value="1e15"/>\n')
            f.write("   </time>\n")
            f.write("   <processing>\n")
            f.write(f'      <lanechange.duration value="{self.road_params["lane_change_duration"]}"/>\n')
            f.write(f'      <lanechange.overtake-right value="{self.road_params["overtake_right"]}"/>\n')
            f.write(
                '      <emergencydecel.warning-threshold value="'
                f'{self.road_params["emergency_decel_warn_threshold"]}"/>\n'
            )
            f.write(f'      <collision.action value="{self.road_params["collision_action"]}"/>\n')
            f.write(f'      <no-step-log value="{self.road_params["no_display_step"]}"/>\n')
            f.write("   </processing>\n")
            f.write("</configuration>\n")

    def _write_gui_settings(self):
        with open(self._path(".settings.xml"), "w") as f:
            f.write("<viewsettings>\n")
            vx, vy = self.road_params["view_position"]
            f.write(f'   <viewport x="{vx}" y="{vy}" zoom="{self.road_params["zoom"]}"/>\n')
            f.write(f'   <delay value="{self.road_params["view_delay"]}"/>\n')
            f.write('   <scheme name="real world"/>\n')
            f.write("</viewsettings>\n")

    def create_road(self):
        self._write_nodes()
        self._write_edges()
        self._write_routes()
        self._write_config()
        self._write_gui_settings()

        netconvert_name = "netconvert.exe" if os.name == "nt" else "netconvert"
        netconvert = os.path.join(self.sumo_home, "bin", netconvert_name)
        result = subprocess.run(
            [
                netconvert,
                "--node-files", self._path(".nod.xml"),
                "--edge-files", self._path(".edg.xml"),
                "--output-file", self._path(".net.xml"),
                "--opposites.guess=true",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"netconvert failed:\n{result.stdout}\n{result.stderr}")

    @property
    def sumocfg_path(self):
        return self._path(".sumocfg")
