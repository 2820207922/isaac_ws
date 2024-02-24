from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim

from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.sensor import _sensor
import omni.kit.commands

from gymnasium import spaces
import numpy as np
import torch
import math

class BalanceTask(BaseTask):
    def __init__(
        self,
        name,
        offset=None
    ) -> None:
        # print("running: __init__")
        # task-specific parameters
        self._robot_default_position = [0.0, 0.0, 0.5]
        self._robot_default_orientation = [1.0, 0.0, 0.0, 0.0]
        self._pos_limit = 1.221731
        self._vel_limit = 1.0
        self._effort_limit = 10.0
        self._angle_target = 0.0
        self._angel_limit = 1.221731
        self._height_target = 0.16
        self._height_limit = 0.45

        # values used for defining RL buffers
        self._num_observations = 12
        self._num_actions = 6
        self._device = "cpu"
        self.num_envs = 1

        # a few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(
            np.ones(self._num_actions, dtype=np.float32) * -1.0, np.ones(self._num_actions, dtype=np.float32) * 1.0
        )
        self.observation_space = spaces.Box(
            np.ones(self._num_observations, dtype=np.float32) * -np.Inf,
            np.ones(self._num_observations, dtype=np.float32) * np.Inf,
        )


        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def set_up_scene(self, scene) -> None:
        # print("running: set_up_scene")
        # retrieve file path for the Cartpole USD file
        usd_path = "/root/isaac_ws/balance_infantry/model/balance_infantry.usd"
        # add the Cartpole USD to our stage
        # create_prim(prim_path="/World", prim_type="Xform", position=self._robot_default_position)
        add_reference_to_stage(usd_path, "/World")

        # Get stage handle
        # stage = omni.usd.get_context().get_stage()
        # if not stage:
        #     print("Stage could not be used.")
        # else:
        #     for prim in stage.Traverse():
        #         prim_path = prim.GetPath()
        #         prim_type = prim.GetTypeName()
        #         print(f"prim_path: {prim_path}, prim_type: {prim_type}")

        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        self._robots = ArticulationView(prim_paths_expr="/World/balance_infantry/base_link*", name="robot_view")
        self._imu_reader = _sensor.acquire_imu_sensor_interface()

        # add Cartpole ArticulationView and ground plane to the Scene
        scene.add(self._robots)
        scene.add_default_ground_plane()

        # set default camera viewport position and target
        self.set_initial_camera_params()

    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def post_reset(self):
        # print("running: post_reset")
        self._base_link_idx = self._robots.get_body_index("base_link")
        self._joint1_idx = self._robots.get_dof_index("joint1")
        self._joint2_idx = self._robots.get_dof_index("joint2")
        self._joint6_idx = self._robots.get_dof_index("joint6")
        self._joint7_idx = self._robots.get_dof_index("joint7")
        self._joint4_idx = self._robots.get_dof_index("joint4")
        self._joint9_idx = self._robots.get_dof_index("joint9")
        # print("base_link_idx: ", self._base_link_idx)
        # print("joint1_idx: ", self._joint1_idx)
        # print("joint2_idx: ", self._joint2_idx)
        # print("joint4_idx: ", self._joint4_idx)
        # print("joint6_idx: ", self._joint6_idx)
        # print("joint7_idx: ", self._joint7_idx)
        # print("joint9_idx: ", self._joint9_idx)
        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset(indices)
    
    def reset(self, env_ids=None):
        # print("running: reset")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        positions = torch.zeros((num_resets, 3), device=self._device)
        orientations = torch.zeros((num_resets, 4), device=self._device)
        orientations[:, 0] = 1.0

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, 4), device=self._device)
        # dof_pos[:, 0] = self._pos_limit * torch.rand(num_resets, device=self._device)
        # dof_pos[:, 1] = - self._pos_limit * torch.rand(num_resets, device=self._device)
        # dof_pos[:, 2] = - self._pos_limit * torch.rand(num_resets, device=self._device)
        # dof_pos[:, 3] = self._pos_limit * torch.rand(num_resets, device=self._device)

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, 6), device=self._device)
        # dof_vel[:, 0] = self._vel_limit * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_vel[:, 1] = self._vel_limit * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_vel[:, 2] = self._vel_limit * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_vel[:, 3] = self._vel_limit * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_vel[:, 4] = self._vel_limit * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_vel[:, 5] = self._vel_limit * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._robots.post_reset()
        # self._robots.set_world_poses(positions, orientations, indices=indices)
        # self._robots.set_joint_positions(dof_pos, indices=indices, joint_indices=torch.tensor([self._joint1_idx, self._joint2_idx, self._joint6_idx, self._joint7_idx]))
        # self._robots.set_joint_velocities(dof_vel, indices=indices, joint_indices=torch.tensor([self._joint1_idx, self._joint2_idx, self._joint6_idx, self._joint7_idx, self._joint4_idx, self._joint9_idx]))
        
        # print("self._robots.get_world_poses(): ", self._robots.get_world_poses())
        # bookkeeping
        self.resets[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        # print("running: pre_physics_step")
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        actions = torch.tensor(actions)
        # print("actions: ", actions)

        forces = torch.zeros((self._robots.count, 6), dtype=torch.float32, device=self._device)
        forces[:, 0] = self._effort_limit * actions[0]
        forces[:, 1] = self._effort_limit * actions[1]
        forces[:, 2] = self._effort_limit * actions[2]
        forces[:, 3] = self._effort_limit * actions[3]
        forces[:, 4] = self._effort_limit * actions[4]
        forces[:, 5] = self._effort_limit * actions[5]

        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
        self._robots.set_joint_efforts(forces, indices=indices, joint_indices=torch.tensor([self._joint1_idx, self._joint2_idx, self._joint6_idx, self._joint7_idx, self._joint4_idx, self._joint9_idx]))

    def get_observations(self):
        # print("running: get_observations")
        # imu_data = self._imu_reader.get_sensor_reading(self._robots.prim_paths[0] + "/imu_sensor", use_latest_data = True, read_gravity = False)
        # print("imu_data.is_valid: ", imu_data.is_valid)
        # print("imu_data.orientation: ", imu_data.orientation)
        positions, orientations = self._robots.get_world_poses()
        if torch.isnan(orientations).any():
            self.reset()
            return
        # print("positions: ", positions)
        # print("orientations: ", orientations)
        angle = self.quaternion_to_euler_zxy(orientations)
        # print("angle: ", angle * 180 / math.pi)

        # collect joint positions and velocities for observation
        dof_pos = self._robots.get_joint_positions()
        dof_vel = self._robots.get_joint_velocities()

        joint1_pos = dof_pos[:, self._joint1_idx]
        joint1_vel = dof_vel[:, self._joint1_idx]
        joint2_pos = dof_pos[:, self._joint2_idx]
        joint2_vel = dof_vel[:, self._joint2_idx]
        joint6_pos = dof_pos[:, self._joint6_idx]
        joint6_vel = dof_vel[:, self._joint6_idx]
        joint7_pos = dof_pos[:, self._joint7_idx]
        joint7_vel = dof_vel[:, self._joint7_idx]

        joint4_vel = dof_vel[:, self._joint4_idx]
        joint9_vel = dof_vel[:, self._joint9_idx]

        self.obs[:, 0] = angle[:, 1]
        self.obs[:, 1] = angle[:, 2]

        self.obs[:, 2] = joint1_pos
        self.obs[:, 3] = joint2_pos
        self.obs[:, 4] = joint6_pos
        self.obs[:, 5] = joint7_pos
        self.obs[:, 6] = joint1_vel
        self.obs[:, 7] = joint2_vel
        self.obs[:, 8] = joint6_vel
        self.obs[:, 9] = joint7_vel

        self.obs[:, 10] = joint4_vel
        self.obs[:, 11] = joint9_vel

        # print("obs: ", self.obs)

        return self.obs
    
    def quaternion_to_euler_zxy(self, q):
        # q = torch.tensor(q)
        quat = torch.zeros((self._robots.count, 4), dtype=torch.float32, device=self._device)
        angle = torch.zeros((self._robots.count, 3), dtype=torch.float32, device=self._device)
        quat[:, 0] = q[:, 0]
        quat[:, 1] = q[:, 1]
        quat[:, 2] = q[:, 2]
        quat[:, 3] = q[:, 3]

        angle[:, 1] = torch.atan2(2.0 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]), 1.0 - 2.0 * (quat[:, 1] * quat[:, 1] + quat[:, 2] * quat[:, 2]))
        angle[:, 2] = torch.asin(torch.clamp(2.0 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), min=-1.0, max=1.0))
        angle[:, 0] = torch.atan2(2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]), 1.0 - 2.0 * (quat[:, 2] * quat[:, 2] + quat[:, 3] * quat[:, 3]))

        # angle = angle * 180 / math.pi

        return angle

    def calculate_metrics(self) -> None:
        # print("running: calculate_metrics")

        roll_x = self.obs[:, 0]
        pitch_y = self.obs[:, 1]

        joint1_pos = self.obs[:, 2]
        joint2_pos = self.obs[:, 3]
        joint6_pos = self.obs[:, 4]
        joint7_pos = self.obs[:, 5]
        joint1_vel = self.obs[:, 6]
        joint2_vel = self.obs[:, 7]
        joint6_vel = self.obs[:, 8]
        joint7_vel = self.obs[:, 9]

        joint4_vel = self.obs[:, 10]
        joint9_vel = self.obs[:, 11]

        left_height = self.calc_height(joint1_pos, joint2_pos)
        right_height = self.calc_height(joint6_pos, joint7_pos)
        
        reward = 1.0 - 2.0 * torch.abs(roll_x / self._angel_limit) - 2.0 * torch.abs(pitch_y / self._angel_limit) - 1.0 * torch.abs((self._height_target - (left_height + right_height) / 2) / self._height_limit)

        return reward.item()

    def calc_height(self, a, b):
        l1 = 0.075
        l2 = 0.15
        l3 = 0.27
        d = 15 * math.pi / 180

        p1 = torch.zeros((self._robots.count, 2), dtype=torch.float32, device=self._device)
        p2 = torch.zeros((self._robots.count, 2), dtype=torch.float32, device=self._device)
        p3 = torch.zeros((self._robots.count, 2), dtype=torch.float32, device=self._device)
        res = torch.zeros((self._robots.count, 1), dtype=torch.float32, device=self._device)
        
        a = torch.abs(a)
        b = torch.abs(b)

        p1[:, 0] = -l2 * torch.cos(a - d) - l1
        p1[:, 1] = l2 * torch.sin(a - d)
        p2[:, 0] = l2 * torch.cos(b - d) + l1
        p2[:, 1] = l2 * torch.sin(b - d)
        p3[:, 0] = (p1[:, 0] + p2[:, 0]) / 2
        p3[:, 1] = (p1[:, 1] + p2[:, 1]) / 2
        res[:, 0] = l3 * l3 - torch.pow(p2[:, 0] - p3[:, 0], 2) - torch.pow(p2[:, 1] - p3[:, 1], 2)
        res[:, 0] = torch.sqrt(res[:, 0] * torch.pow(p2[:, 0]- p1[:, 0], 2) / (torch.pow(p2[:, 0]- p1[:, 0], 2) + torch.pow(p2[:, 1]- p1[:, 1], 2)))

        return res


    def is_done(self) -> None:
        # print("running: is_done")
        roll_x = self.obs[:, 0]
        pitch_y = self.obs[:, 1]

        joint1_pos = self.obs[:, 2]
        joint2_pos = self.obs[:, 3]
        joint6_pos = self.obs[:, 4]
        joint7_pos = self.obs[:, 5]
        joint1_vel = self.obs[:, 6]
        joint2_vel = self.obs[:, 7]
        joint6_vel = self.obs[:, 8]
        joint7_vel = self.obs[:, 9]

        joint4_vel = self.obs[:, 10]
        joint9_vel = self.obs[:, 11]

        # reset the robot if cart has reached reset_dist or pole is too far from upright
        resets = torch.where(torch.abs(roll_x) > self._angel_limit, 1, 0)
        resets = torch.where(torch.abs(pitch_y) > self._angel_limit, 1, resets)

        resets = torch.where(joint1_pos > self._pos_limit, 1, resets)
        resets = torch.where(joint1_pos < 0, 1, resets)
        resets = torch.where(joint2_pos > 0, 1, resets)
        resets = torch.where(joint2_pos < -self._pos_limit, 1, resets)
        resets = torch.where(joint6_pos > 0, 1, resets)
        resets = torch.where(joint6_pos < -self._pos_limit, 1, resets)
        resets = torch.where(joint7_pos > self._pos_limit, 1, resets)
        resets = torch.where(joint7_pos < 0, 1, resets)

        resets = torch.where(torch.abs(joint1_vel) > self._vel_limit, 1, resets)
        resets = torch.where(torch.abs(joint2_vel) > self._vel_limit, 1, resets)
        resets = torch.where(torch.abs(joint6_vel) > self._vel_limit, 1, resets)
        resets = torch.where(torch.abs(joint7_vel) > self._vel_limit, 1, resets)

        resets = torch.where(torch.abs(joint4_vel) > self._vel_limit, 1, resets)
        resets = torch.where(torch.abs(joint9_vel) > self._vel_limit, 1, resets)

        self.resets = resets
        # print("resets: ", resets.item())

        return resets.item()
