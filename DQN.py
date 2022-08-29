import sys
sys.path.insert(1,'D:\\Vscode\\SEDDPG\\Highway-RL0301\\highway-rl\\stable-baselines3-master')
sys.path.insert(1,'D:\\VsCode\\SEDDPG\\Highway-RL0301\\highway-rl\\highway-env\\highway-env-master')

import torch
import numpy as np
import gym
import time
import os
import datetime
import sys
from gym import wrappers
import xlsxwriter
from tensorboardX import SummaryWriter

import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# 设定环境信息
env_config = {
"id": "highway-v0",
"import_module": "highway_env",
"observation": {
    "type": "Kinematics",  # 观测类型
    "features": [  # 观测的内容
        "presence",
        "x",
        "y",
        "vx",
        "vy",
        "sin_h"
    ],
    "vehicles_count": 4,  # 观测的车辆数
    "absolute": False,  # 观测内容是否为绝对值
    "order": "sorted",  # Order of observed vehicles. Values: sorted, shuffled
    "normalize": True,  # 是否进行归一化
    "clip": True,  # 观测值是否进行裁剪
    "see_behind": False,  # 观测车辆是否包含后车（观测后车的判断里就算这个不满足，在后方一定范围内的车辆也可以被观测到）
    "observe_intentions": False  # Observe the destinations of other vehicles
},

"action": {  # 自车动作类型
    "type": "DiscreteMetaAction",
    "longitudinal": True,  # 是否启用纵向控制
    "lateral": True
},
"lanes_count": 3,  # 车道数
"vehicles_count": 30,  # 车辆数
"controlled_vehicles": 1,
"initial_lane_id": None,  # 自车初始生成的车道
"duration": 40,  # 单次episode的最多step  测试时为800，训练时改成200
"ego_spacing": 2,  # 生成车辆时会用到
"vehicles_density": 1,  # 生成车辆时会用到
"initial_spacing": 0.5,  # 初始车辆间距
"collision_reward": -15,  # 碰撞奖励
"reward_speed_range": [17, 33],  # 奖励速度范围
"simulation_frequency": 20,  # [Hz]         # 环境模拟频率  20
"policy_frequency": 1,  # [Hz]              #
"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # 他车动作类型
"screen_width": 800,  # [px]                # 画面长度
"screen_height": 150,  # [px]
"centering_position": [0.3, 0.5],  # 画面中心位置
"scaling": 5.5,
"show_trajectories": False,
"display_record": True,  # 是否启用Monitor进行录像
"render_agent": True,
"offscreen_rendering": True,  # 程序运行时是否显示画面
"manual_control": False,
"real_time_rendering": False,  # 是否实时显示，false时画面是加速过的
"offroad_terminal": True  # 是否在跑出时终止本次episode
}

if __name__ == '__main__':
    TRAIN = False 
    time_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    save_path = 'out/HighwayEnv/DQN_SB/Train_{}_{}/DQNAgent'.format(time_now, os.getpid())
    excel = xlsxwriter.Workbook('out/HighwayEnv/DQN_SB/Test_{}_{}/record.xlsx'.format(time_now, os.getpid()))
    sheet = excel.add_worksheet()
    sheet.write(0, 0, 'episode')
    sheet.write(0, 1, 'mean_reward')
    sheet.write(0, 2, 'mean_right_lane_reward')
    sheet.write(0, 3, 'mean_high_speed_reward')
    sheet.write(0, 4, 'mean_comfort_reward')
    sheet.write(0, 5, 'mean_lane_center_reward')
    sheet.write(0, 6, 'mean_lateral_acc')
    sheet.write(0, 7, 'mean_speed')
    sheet.write(0, 8, 'ep_reward')
    sheet.write(0, 9, 'ep_step')
    sheet.write(0, 10, 'travel_distance')
    sheet.write(0, 11, 'total_step')
    sheet.write(0, 12, 'Run_time')
    if TRAIN:
        seed = 2
        log_path = 'out/HighwayEnv/DQN_SB/Train_{}_{}'.format(time_now, os.getpid())
    else:
        seed = 3
        log_path = 'out/HighwayEnv/DQN_SB/Test_{}_{}'.format(time_now, os.getpid())


    env = gym.make(env_config['id'])
    env.unwrapped.configure(env_config)
    env.reset()  # Reset the environment to ensure configuration is applied
    env = wrappers.Monitor(env, log_path, video_callable=(None if env_config["display_record"] else False))  # Monitor要放在env.reset后面,env_config中的action设置才会生效
    # seed设置
    env.seed(seed)
    if TRAIN:
        model = DQN('MlpPolicy', env,
                        policy_kwargs=dict(net_arch=[256, 256]),
                        learning_rate=5e-4,
                        buffer_size=1000000,
                        learning_starts=200,
                        batch_size=256,
                        gamma=0.9,
                        train_freq=1,
                        gradient_steps=1,
                        target_update_interval=50,
                        verbose=1,
                        exploration_fraction=0.6,
                        tensorboard_log=log_path)
        model.learn(total_timesteps=int(8e5))
        model.save(save_path)
        # load_path = save_path
        # del model
    else:
        load_path = 'D:\Vscode\SEDDPG\Highway-RL0301\highway-rl\out\HighwayEnv\DQN_SB\Train_20220701-220619_56032\DQNAgent'
        model = DQN.load(load_path, env=env)
    # testing
    total_step = 0
    start_time = time.time()
    for episode in range(120):
        # 每个回合初始化统计数据
        ep_reward = 0
        ep_right_lane_reward = 0
        ep_high_speed_reward = 0
        ep_comfort_reward = 0
        ep_lateral_acc = 0
        ep_lane_center_reward = 0
        ep_speed = 0
        ep_step = 0
        init_position = env.vehicle.position[0]
        obs, done = env.reset(), False
        env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
        step = 0
        while not done:
            if env_config["display_record"]:
                env.render()
            action, _ = model.predict(obs, deterministic=True)
            step = step + 1
            obs, reward, done, info = env.step(action)
            # 每个回合更新数据
            if reward < 0:
                reward = reward/20
            ep_reward += reward
            ep_right_lane_reward += info["right_lane_reward"]
            ep_high_speed_reward += info["high_speed_reward"]
            ep_comfort_reward += info["comfort_reward"]
            ep_lane_center_reward += info["lane_center_reward"]
            ep_lateral_acc += np.abs(np.sin(env.vehicle.heading + env.vehicle.beta))
            ep_speed += info["speed"]
            ep_step += 1
            total_step += 1
        # 回合结束统计平均数据
        episode += 1
        final_position = env.vehicle.position[0]  # episode结束时自车的x坐标
        travel_distance = final_position - init_position
        mean_reward = ep_reward / 40 #ep_step
        mean_right_lane_reward = ep_right_lane_reward / ep_step
        mean_high_speed_reward = ep_high_speed_reward / ep_step
        mean_comfort_reward = ep_comfort_reward / ep_step
        mean_lane_center_reward = ep_lane_center_reward / ep_step
        mean_lateral_acc = ep_lateral_acc / ep_step
        mean_speed = ep_speed / ep_step
        # 写入excel
        sheet.write(episode, 0, episode)
        sheet.write(episode, 1, mean_reward)
        sheet.write(episode, 2, mean_right_lane_reward)
        sheet.write(episode, 3, mean_high_speed_reward)
        sheet.write(episode, 4, mean_comfort_reward)
        sheet.write(episode, 5, mean_lane_center_reward)
        sheet.write(episode, 6, mean_lateral_acc)
        sheet.write(episode, 7, mean_speed)
        sheet.write(episode, 8, ep_reward)
        sheet.write(episode, 9, ep_step)
        sheet.write(episode, 10, travel_distance)
        sheet.write(episode, 11, total_step)
        print("episode:{} mean_reward:{:.2f} mean_speed:{:.2f} mean_lateral_acc:{:.4f} total_reward:{:.2f} step:{} total_step:{} time_cost: {}".format(episode, mean_reward, mean_speed, mean_lateral_acc, ep_reward, ep_step, total_step, time.time()-start_time))
    env.close()
    excel.close()
