# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# In this example, please drag the cube along the arm and see how the effort measurement from the effort sensor changes

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys

import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor.scripts.effort_sensor import EffortSensor
from pxr import UsdPhysics

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60, rendering_dt=1.0 / 60)
my_world.scene.add_default_ground_plane(z_position=-1)

asset_path = assets_root_path + "/Isaac/Robots/Simple/simple_articulation.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/Articulation")
arm_joint = "/Articulation/Arm/RevoluteJoint"
prim = get_prim_at_path(arm_joint)
joint = UsdPhysics.RevoluteJoint(prim)
joint.CreateAxisAttr("Y")

DynamicCuboid(
    prim_path="/World/Cube",
    name="cube_1",
    position=np.array([1.5, 0, 0.2]),
    color=np.array([255, 0, 0]),
    size=0.1,
    mass=1,
)

my_world.reset()
effort_sensor = EffortSensor(prim_path=arm_joint)

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        reading = effort_sensor.get_sensor_reading()
        print(f"Sensor Time: {reading.time}   Value: {reading.value}   Validity: {reading.is_valid}")

        if my_world.current_time_step_index == 0:
            my_world.reset()

simulation_app.close()