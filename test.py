#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import argparse

import numpy as np

import gym
import gym_carla
import carla

def main():

  parser = argparse.ArgumentParser(description="Autoencoder training.")
  parser.add_argument("--path", default="/mnt/storage2/carla-test-1/", type=str,help="Save dataset path")
  parser.add_argument("--folder", default="test", type=str,help="Save folder")
  args = parser.parse_args()


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
    'max_time_episode': 500,  # maximum timesteps per episode
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
    #'folder': "/mnt/storage2/carla-test-1/Log1"
  }

  params['folder'] = os.path.join(args.path, args.folder)

  

  folders = [f"Log{s}" for s in np.arange(75,101)]

  if not os.path.exists(args.path):
    os.mkdir(args.path)

  log_idx = 0
  params['folder'] = os.path.join(args.path, folders[log_idx])
  print(params['folder'])

  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)
  obs = env.reset()

  while True:
    action = [2.0, 0.0]
    obs,r,done,info = env.step(action)

    if done:
      obs = env.reset()
      log_idx += 1
      folder = os.path.join(args.path, folders[log_idx])
      env.set_folder(folder)
      print(folder)

if __name__ == '__main__':
  main()
