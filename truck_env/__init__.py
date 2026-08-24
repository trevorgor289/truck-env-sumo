"""
truck_env: a Gymnasium environment for tactical decision-making of an
autonomous truck in SUMO, modeled on deepthi-pathare/Autonomous-truck-sumo-gym-env's
`new_architecture` (semantic RL actions + low-level longitudinal/lateral
controllers), rewritten against SUMO 1.27.1 instead of the reference repo's
pinned 1.15.0.

This must run before any sibling module does `import traci` / `import sumolib`.
"""

import os
import sys

if "SUMO_HOME" not in os.environ:
    raise RuntimeError("Please set the SUMO_HOME environment variable before importing truck_env.")

_tools = os.path.join(os.environ["SUMO_HOME"], "tools")
if _tools not in sys.path:
    sys.path.append(_tools)
