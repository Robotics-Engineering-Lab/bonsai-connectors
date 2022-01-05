"""
UR10 robot arm Environment with Robotiq Gripper for REACH-skill
License: MIT- license
"""

from numpy.core.fromnumeric import take
import pybullet
import pybullet_data
import gym
import numpy as np
import sys, os

class UR10(gym.Env):
    observation_space = gym.spaces.Dict(dict(
        desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        observation=gym.spaces.Box(-np.inf, np.inf, shape=(9,), dtype='float32'),
    ))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
    position_bounds = [(0.2, 1), (-0.4, 0.4), (0.64, 1)]
    joint_type = ['REVOLUTE', 'PRISMATIC', 'SPHERICAL', 'PLANAR', 'FIXED']
    initial_joint_values = np.array((1.2, 0.0, 0.8))
    distance_threshold = 0.05

    def __init__(self, is_train=False, is_dense=False):
        self.connect(is_train)
        self.reward = 0
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.is_dense = is_dense
    
    def connect(self, is_train):
        if is_train:
            pybullet.connect(pybullet.DIRECT)
        else:
            pybullet.connect(pybullet.GUI)
    def start_log_video(self, filename):
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, filename)

    def stop_log_video(self):
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)

    def render(self, mode='human'):
        pass

    def __del__(self):
        pybullet.disconnect()

    def _rescale(self, values, bounds):
        result = np.zeros_like(values)
        for i, (value, (lower_bound, upper_bound)) in enumerate(zip(values, bounds)):
            result[i] = (value + 1) / 2 * (upper_bound - lower_bound) + lower_bound
        return result

    
    def compute_state(self):
        # gripper state
        gripper_position, gripper_orientation, _, _, _, _, gripper_velocity, gripper_angular_velocity = \
            pybullet.getLinkState(self.robot, linkIndex = 7, computeLinkVelocity=True)    
        gripper_position = np.asarray(gripper_position)
        gripper_orientation = pybullet.getEulerFromQuaternion(gripper_orientation)

        # target state
        target_position, target_orientation = pybullet.getBasePositionAndOrientation(self.target)
        target_position = np.asarray(target_position)
        target_orientation = pybullet.getEulerFromQuaternion(target_orientation)

        # state vector
        gripper_observation = np.concatenate(
            (
                gripper_position,
                #gripper_orientation,
                #gripper_velocity,
                #gripper_angular_velocity
            )
        )
        target_observation = np.concatenate(
            (
                target_position,
                #target_orientation,
                #target_velocity,
                #target_angular_velocity
            )
        )
        extended_observation = np.concatenate(
            (
                gripper_position - target_position,
            )
        )
        state = np.concatenate((gripper_observation, target_observation, extended_observation))
        return {'observation': np.array(state, np.float), 'desired_goal': np.asarray(target_position, np.float), 'achieved_goal': np.asarray(gripper_position, np.float)}

    def compute_reward(self, state):
        rew = -np.linalg.norm(state['achieved_goal'] - state['desired_goal'])
        return rew

    def step(self, action):
        self.step_id += 1
        self.joint_values += np.array(action[:3]) * 0.1
        self.joint_values = np.clip(self.joint_values, -1, 1)
        target_pos = self._rescale(self.joint_values, self.position_bounds)
        self.move_hand(target_pos)

        pybullet.stepSimulation()
        state = self.compute_state()
        info = self.compute_info(action, state)
        return state, self.compute_reward(state), self.is_done(), info

    def is_done(self):
        return self.step_id == 500

    def compute_info(self, last_action, state):
        distance = np.linalg.norm(state['achieved_goal'] - np.array(state['desired_goal']))
        return {
            'is_success': distance < self.distance_threshold,
            'gripper_pos': 0.0,
            'last_action': last_action
        }
    

    def move_hand(self, target_position, orientation = (np.pi/2, 0, np.pi / 2), gripper_value = 0):
        ee_index = 7
        joint_poses = pybullet.calculateInverseKinematics(
            self.robot,
            #self.links_fixed['ee_joint'], 
            ee_index,
            target_position,
            targetOrientation = pybullet.getQuaternionFromEuler(orientation),
            maxNumIterations=1000,
            residualThreshold=.01
        )
        for joint, pos in zip(self.joints, joint_poses):
            pybullet.setJointMotorControl2(
                self.robot, joint['jointID'],
                pybullet.POSITION_CONTROL,
                targetPosition=pos,
            )
        
    def reset(self):
        self._reset_world()
        pybullet.stepSimulation()
        state = self.compute_state()
        return state

    def _reset_world(self):
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, -9.8)
        self.planeId = pybullet.loadURDF('plane.urdf')
        self.table = pybullet.loadURDF('table/table.urdf', globalScaling=1, basePosition=[0.5, 0, 0])
        self.robot, self.joints, self.joints_rev, self.joints_fix = self.spawn_robot('3d_models_main/ur10_urdf/ur10/ur10_exp.urdf', pos=(0, 0, 1), angle=(0, 0, 0)) 

        x_pos_targ = np.random.uniform(0.5, 0.9)
        y_pos_targ = np.random.uniform(-0.3, 0.3)
        z_pos_targ = np.random.uniform(0.675, 1)
        self.target_origin_pos = np.asarray((x_pos_targ, y_pos_targ, z_pos_targ))
        self.target = pybullet.loadURDF('3d_models_main/sphere/sphere.urdf', self.target_origin_pos, pybullet.getQuaternionFromEuler([0, 0, np.pi / 2]))
        
        self.step_id = 0
        self.joint_values = self.initial_joint_values
        self.gripper_value = -1

    def spawn_robot(self, urdf_path, pos, angle, ee_pos = None, ee_orientation = None):
        """
        Spawn UR_10 robot \n
        urdf_path - path to urdf model\n
        pos - position of base_link\n
        angle - angle of base_link\n
        ee_pos - statr position of end effector\n
        Returns:\n
        -pybullet id\n
        -joints\n
        -links\n
        -fixed links\n
        """
        robot = pybullet.loadURDF(urdf_path, pos, pybullet.getQuaternionFromEuler(angle))
        joints = []
        joints_rev = {}
        joints_fix = {}

        for joint_id in range(pybullet.getNumJoints(robot)):
            info = pybullet.getJointInfo(robot, joint_id)
            data = {
                'jointID': info[0],
                'jointName': info[1].decode('utf-8'),
                'jointType': self.joint_type[info[2]],
                'jointLowerLimit': info[8],
                'jointUpperLimit': info[9],
                'jointMaxForce': info[10],
                'jointMaxVelocity': info[11]
            }
            if data['jointType'] != 'FIXED':
                joints.append(data)
                joints_rev[data['jointName']] = joint_id
            else:
                joints_fix[data['jointName']] = joint_id
        joint_poses = [0] * 6
        joint_poses[1] = -0.5
        joint_poses[2] = 1
        
        for joint_id in range(6):
            pybullet.setJointMotorControl2(
                robot, joints[joint_id]['jointID'],
                pybullet.POSITION_CONTROL,
                targetPosition=joint_poses[joint_id],
            )
        for i in range(5):
            pybullet.stepSimulation()
        return robot, joints, joints_rev, joints_fix