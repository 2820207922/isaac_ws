#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp

# This sample enables a livestream server to connect to when running headless
KIT_CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "renderer": "RayTracedLighting",
    "display_options": 3286,  # Set display options to show default grid
}
kit = SimulationApp(KIT_CONFIG)


from omni.isaac.core.articulations import Articulation
from omni.isaac.sensor import IMUSensor
from omni.importer.urdf import _urdf
from omni.isaac.dynamic_control import _dynamic_control
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics, Tf
import omni.kit.commands
import numpy as np
import math

def quaternion_to_euler(q):
    """
    Convert a quaternion into euler angles (yaw, roll, pitch)
    Quaternion format: [w, x, y, z]
    Euler angles order: yaw (Z), roll (X), pitch (Y)
    """
    # Extract the values from quaternion
    w, x, y, z = q
    
    # Pre-calculate common terms
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x**2 + y**2)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y**2 + z**2)
    yaw_z = math.atan2(t3, t4)
    
    return yaw_z, roll_x, pitch_y  # Order: yaw, roll, pitch



# Acquire the URDF extension interface
urdf_interface = _urdf.acquire_urdf_interface()
# Set the settings in the import config
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.fix_base = False
import_config.import_inertia_tensor = True
import_config.distance_scale = 1.0
import_config.density = 0.0
import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_VELOCITY
import_config.default_drive_strength = 0.0
import_config.default_position_drive_damping = 0.0
import_config.convex_decomp = False
import_config.self_collision = False
import_config.create_physics_scene = True
import_config.make_default_prim = False

# Get path to extension data:
URDF_PATH = "balance_infantry/model.urdf"
DEST_PATH = "balance_infantry/model/model.usd"
# Import URDF, stage_path contains the path the path to the usd prim in the stage.
status, stage_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
    get_articulation_root=True,
)
# Get stage handle
stage = omni.usd.get_context().get_stage()

# Enable physics
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
# Set gravity
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(9.81)
# Set solver settings
PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
physxSceneAPI.CreateEnableCCDAttr(True)
physxSceneAPI.CreateEnableStabilizationAttr(True)
physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
physxSceneAPI.CreateSolverTypeAttr("TGS")
# Set limit
LOWER_LIMIT_ANGLE = 0
UPPER_LIMIT_ANGLE = 70
left_front_joint_prim = UsdPhysics.RevoluteJoint.Get(stage, "/balance_infantry/base_link/joint1")
left_front_joint_prim.GetLowerLimitAttr().Set(LOWER_LIMIT_ANGLE)
left_front_joint_prim.GetUpperLimitAttr().Set(UPPER_LIMIT_ANGLE)
left_back_joint_prim = UsdPhysics.RevoluteJoint.Get(stage, "/balance_infantry/base_link/joint2")
left_back_joint_prim.GetLowerLimitAttr().Set(-UPPER_LIMIT_ANGLE)
left_back_joint_prim.GetUpperLimitAttr().Set(-LOWER_LIMIT_ANGLE)
right_front_joint_prim = UsdPhysics.RevoluteJoint.Get(stage, "/balance_infantry/base_link/joint7")
right_front_joint_prim.GetLowerLimitAttr().Set(LOWER_LIMIT_ANGLE)
right_front_joint_prim.GetUpperLimitAttr().Set(UPPER_LIMIT_ANGLE)
right_back_joint_prim = UsdPhysics.RevoluteJoint.Get(stage, "/balance_infantry/base_link/joint6")
right_back_joint_prim.GetLowerLimitAttr().Set(-UPPER_LIMIT_ANGLE)
right_back_joint_prim.GetUpperLimitAttr().Set(-LOWER_LIMIT_ANGLE)

# Set constraint
left_wheel_link = stage.GetPrimAtPath("/balance_infantry/left_wheel_link")
left_hole_link = stage.GetPrimAtPath("/balance_infantry/left_hole_link")
left_constraint = UsdPhysics.RevoluteJoint.Define(stage, "/balance_infantry/base_link/left_constraint")
left_constraint.CreateBody0Rel().SetTargets([left_wheel_link.GetPath()])
left_constraint.CreateBody1Rel().SetTargets([left_hole_link.GetPath()])
left_constraint.CreateAxisAttr().Set("X")
left_constraint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
left_constraint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
left_constraint.CreateExcludeFromArticulationAttr().Set(True)
right_wheel_link = stage.GetPrimAtPath("/balance_infantry/right_wheel_link")
right_hole_link = stage.GetPrimAtPath("/balance_infantry/right_hole_link")
right_constraint = UsdPhysics.RevoluteJoint.Define(stage, "/balance_infantry/base_link/right_constraint")
right_constraint.CreateBody0Rel().SetTargets([right_wheel_link.GetPath()])
right_constraint.CreateBody1Rel().SetTargets([right_hole_link.GetPath()])
right_constraint.CreateAxisAttr().Set("X")
right_constraint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
right_constraint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
right_constraint.CreateExcludeFromArticulationAttr().Set(True)
# Add ground plane
omni.kit.commands.execute(
    "AddGroundPlaneCommand",
    stage=stage,
    planePath="/groundPlane",
    axis="Z",
    size=150.0,
    position=Gf.Vec3f(0, 0, -0.3),
    color=Gf.Vec3f(0.3),
)

# Add lighting
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(500)

# Start simulation
omni.timeline.get_timeline_interface().play()
# perform one simulation step so physics is loaded and dynamic control works.
kit.update()
art = Articulation(prim_path=stage_path)
art.initialize()

if not art.handles_initialized:
    print(f"{stage_path} is not an articulation")
else:
    print(f"Got articulation {stage_path} with handle {art.articulation_handle}")

dc = _dynamic_control.acquire_dynamic_control_interface()
dc.wake_up_articulation(art.articulation_handle)

dof_properties = _dynamic_control.DofProperties()
dof_properties.damping = 0.0
dof_properties.stiffness = 0.0
dof_properties.max_effort = 100.0
dof_properties.max_velocity = 10.0
left_constraint = dc.find_articulation_dof(art.articulation_handle, "left_constraint")
right_constraint = dc.find_articulation_dof(art.articulation_handle, "right_constraint")
dc.set_dof_properties(left_constraint, dof_properties)
dc.set_dof_properties(right_constraint, dof_properties)

dof_properties.damping = 100.0
dof_properties.stiffness = 0.0
dof_properties.max_effort = 100.0
dof_properties.max_velocity = 10.0

left_wheel_joint = dc.find_articulation_dof(art.articulation_handle, "joint4")
right_wheel_joint = dc.find_articulation_dof(art.articulation_handle, "joint9")
dc.set_dof_properties(left_wheel_joint, dof_properties)
dc.set_dof_properties(right_wheel_joint, dof_properties)

left_front_joint = dc.find_articulation_dof(art.articulation_handle, "joint1")
left_back_joint = dc.find_articulation_dof(art.articulation_handle, "joint2")
right_front_joint = dc.find_articulation_dof(art.articulation_handle, "joint7")
right_back_joint = dc.find_articulation_dof(art.articulation_handle, "joint6")
dc.set_dof_properties(left_front_joint, dof_properties)
dc.set_dof_properties(left_back_joint, dof_properties)
dc.set_dof_properties(right_front_joint, dof_properties)
dc.set_dof_properties(right_back_joint, dof_properties)
# Set IMU sensor
# imu_sensor = IMUSensor(
#     prim_path="/balance_infantry/base_link/imu_sensor",
#     name="imu",
#     frequency=100,
#     translation=np.array([0.0, -0.2, 0.1]),
# )
# imu_sensor.initialize()

if not stage:
    print("Stage could not be used.")
else:
    for prim in stage.Traverse():
        prim_path = prim.GetPath()
        prim_type = prim.GetTypeName()

        print(f"prim_path: {prim_path}, prim_type: {prim_type}")

k = 0
torque = 10
# perform simulation
while kit._app.is_running() and not kit.is_exiting():
    # Run in realtime mode, we don't specify the step size
    k = k + 1
    if k // 500 % 2 == 1:
        dc.set_dof_effort(left_front_joint, torque)
        dc.set_dof_effort(left_back_joint, -torque)
        dc.set_dof_effort(right_front_joint, torque)
        dc.set_dof_effort(right_back_joint, -torque)
    else:
        dc.set_dof_effort(left_front_joint, -torque)
        dc.set_dof_effort(left_back_joint, torque)
        dc.set_dof_effort(right_front_joint, -torque)
        dc.set_dof_effort(right_back_joint, torque)

    # imu_data = imu_sensor.get_current_frame()
    # quaternion = imu_data['orientation']
    # # quaternion = [1.0, 0.0, 0.0, 0.0]
    # rotation = quaternion_to_euler(quaternion)
    # euler_angles_deg = np.degrees(rotation)
    # print(f"quaternion: {quaternion}, angle: {euler_angles_deg}")

    kit.update()

# Shutdown and exit
omni.timeline.get_timeline_interface().stop()
kit.close()
