#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import argparse
import time

import numpy as np

import gym
import gym_carla
import carla

from PIL import Image
import os
import csv
import numpy as np   

import json
import cv2

from functools import reduce

from skimage.transform import resize
from skimage.io import imsave

def save_state(obs, path_save="/results"):
    """ Save the state in the FordAV-like format. """

    ts = int(time.time()*1000)  # timestep

    camera_keys =  ['camera_f', 'camera_sl', 'camera_sr', 'camera_r']
    
    # Save camera image
    for key in camera_keys:
      path_save_camera = reduce(os.path.join, [path_save, key, f"{ts}.png"])
      imsave(path_save_camera, obs[key])

      # Temp workaround
      image = cv2.imread(path_save_camera, cv2.IMREAD_GRAYSCALE)
      image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
      imsave(reduce(os.path.join, [path_save, f"{key}_resized_128", f"{ts}.png"]), image)

    # Save semseg camera image
    #imsave(reduce(os.path.join, [path_save, "camera_f_ss", f"{ts}.png"]), obs["semantic"])
    # Save lidar image
    # imsave(reduce(os.path.join, [path_save, "lidar", f"{ts}.png"]), obs["lidar"])
    # Save lidar BEV image
    # imsave(reduce(os.path.join, [path_save, "lidar_bev", f"{ts}.png"]), obs["lidar"])

    # Save IMU data

    # Temporary - for state
    state_dict = {
      "steer": obs['state'][0],
      "delta_yaw": obs['state'][1],
      "lspeed_lon": obs['state'][2],
      "speed": obs['state'][3]
    }
    sensors_dict = obs['imu'].copy()
    sensors_dict.update(state_dict)

    with open(os.path.join(os.path.join(path_save, f"sensors"), f"{ts}.json"), 'w') as f:
      json.dump(sensors_dict, f, indent=4)


def main():

  parser = argparse.ArgumentParser(description="Autoencoder training.")
  parser.add_argument("--path", default="/mnt/storage2/carla-test-0", type=str,help="Save dataset path")
  parser.add_argument("--folder", default="test", type=str,help="Save folder")
  args = parser.parse_args()

  if not os.path.exists(args.path):
    os.mkdir(args.path)

  # parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 60,
    'number_of_walkers': 20,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 509,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': False,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
  }


  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)
  obs = env.reset()
  for _ in range(10):  # wait for zoom
    env.step([2.0, 0.0])
  
  keys = [
    'camera_f', 'camera_sl', 'camera_sr', 'camera_r',
    'camera_f_resized_128', 'camera_sl_resized_128', 'camera_sr_resized_128', 'camera_r_resized_128',
    'camera_f_ss', 
    #'lidar', 'lidar_bev', 'semantic', 
    'sensors'
  ]

  logs = [f"Log_{s}" for s in np.arange(174, 200)]
  li = 0  # Log index
  log_current = os.path.join(args.path, logs[li])
  if not os.path.exists(log_current):
    os.mkdir(log_current)
  # Create subfolders
  for key in keys:
    if not os.path.exists(os.path.join(log_current, key)):
      os.mkdir(os.path.join(log_current, key))

  time_start = time.time()

  while True:

    action = [2.0, 0.0]
    obs,r,done,info = env.step(action)

    save_state(obs, path_save=log_current)

    if done:
      obs = env.reset()
      for _ in range(15):
        env.step(action)
      li +=1

      log_current = os.path.join(args.path, logs[li])
      if not os.path.exists(log_current):
        os.mkdir(log_current)
      for key in keys:
        if not os.path.exists(os.path.join(log_current, key)):
          os.mkdir(os.path.join(log_current, key))

      print(f"Log {li-1} Elapsed {(time.time()-time_start):.2f} seconds")
      time_start = time.time()


if __name__ == '__main__':
  np.set_printoptions(precision=3)
  main()
