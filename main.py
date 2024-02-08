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
from omni.importer.urdf import _urdf
from omni.isaac.dynamic_control import _dynamic_control
from pxr import Gf, PhysxSchema, Sdf, UsdLux, UsdPhysics
import omni.kit.commands

# Acquire the URDF extension interface
urdf_interface = _urdf.acquire_urdf_interface()
# Set the settings in the import config
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.fix_base = True
import_config.import_inertia_tensor = True
import_config.distance_scale = 1
import_config.density = 0.0
import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_VELOCITY
import_config.default_drive_strength = 0
import_config.default_position_drive_damping = 0
import_config.convex_decomp = False
import_config.self_collision = False
import_config.create_physics_scene = True
import_config.make_default_prim = True

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

if not stage:
    print("Stage could not be used.")
else:
    for prim in stage.Traverse():
        prim_path = prim.GetPath()
        prim_type = prim.GetTypeName()

        print(f"prim_path: {prim_path}, prim_type: {prim_type}")

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

# Add ground plane
omni.kit.commands.execute(
    "AddGroundPlaneCommand",
    stage=stage,
    planePath="/groundPlane",
    axis="Z",
    size=150.0,
    position=Gf.Vec3f(0, 0, -0.5),
    color=Gf.Vec3f(0.5),
)

# Add lighting
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(500)

# # Get handle to the Drive API for both wheels
# left_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/balance_infantry/left_back_calf_link/joint4"), "angular")
# right_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/balance_infantry/right_back_calf_link/joint9"), "angular")

# # Set the velocity drive target in degrees/second
# left_wheel_drive.GetTargetVelocityAttr().Set(180)
# right_wheel_drive.GetTargetVelocityAttr().Set(180)

# # Set the drive damping, which controls the strength of the velocity drive
# left_wheel_drive.GetDampingAttr().Set(15000)
# right_wheel_drive.GetDampingAttr().Set(15000)

# # Set the drive stiffness, which controls the strength of the position drive
# # In this case because we want to do velocity control this should be set to zero
# left_wheel_drive.GetStiffnessAttr().Set(0)
# right_wheel_drive.GetStiffnessAttr().Set(0)

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

num_joints = dc.get_articulation_joint_count(art.articulation_handle)
num_dofs = dc.get_articulation_dof_count(art.articulation_handle)
num_bodies = dc.get_articulation_body_count(art.articulation_handle)
print("num_joints = ", num_joints, "num_dofs = ", num_dofs, "num_bodies = ", num_bodies)

dc.wake_up_articulation(art.articulation_handle)
left_wheel_joint = dc.find_articulation_dof(art.articulation_handle, "joint4")
right_wheel_joint = dc.find_articulation_dof(art.articulation_handle, "joint9")

k = 0
# perform simulation
while kit._app.is_running() and not kit.is_exiting():
    # Run in realtime mode, we don't specify the step size
    # k = k + 1
    # print(k//1000 + 1)
    # for i in range(12):
    #     joint_t = dc.find_articulation_dof(art.articulation_handle, "joint" + str(i+1))
    #     if i == k // 1000:
    #         dc.set_dof_effort(joint_t, 50)
    #     else:
    #         dc.set_dof_effort(joint_t, 0)
    dc.set_dof_effort(left_wheel_joint, 5)
    dc.set_dof_effort(right_wheel_joint, -5)
    kit.update()

# Shutdown and exit
omni.timeline.get_timeline_interface().stop()
kit.close()
