"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys
import numpy as np
from collections import namedtuple
from itertools import count
import random
from control import control_robot
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

depth_data = None
rgb_data = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
def read_image():
    rospy.init_node('read_image', anonymous=True)

    rospy.Subscriber('/camera/depth/image_raw', Image, read_depth)
    #rospy.Subscriber('/camera/rgb/image_raw', Image, read_rgb)

    # Allow up to one second to connection
    rospy.sleep(3)
def read_depth(data):
    global depth_data

    # # Convert image to OpenCV format
    try:
        bridge = CvBridge()
        depth_data = bridge.imgmsg_to_cv2(data, "passthrough")
        
    except CvBridgeError as e:
        print(e)

def read_rgb(data):
    global rgb_data

    # # Convert image to OpenCV format
    # try:
    #     bridge = CvBridge()
    #     rgb_data = bridge.imgmsg_to_cv2(data, "bgr8")
    # except CvBridgeError as e:
    #     print(e)

    data_str = str(data)
    idx = data_str.find('[')
    img = data_str[idx+1:-1].split(', ')
    rgb_data = np.array(img, dtype=np.int8)
    rgb_data = rgb_data.reshape(480, 640, 3)
    
class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}


def dqn_learing(
    #env,
    q_func,
    optimizer_spec,
    exploration,
    #stopping_criterion=None,
    replay_buffer_size=1000,
    batch_size=32,
    gamma=0.99,
    learning_starts=1,
    learning_freq=4,
    frame_history_len=1,
    target_update_freq=10000
    ):


    #our own code
    read_image()
    rgb_data=depth_data.reshape(640,480,1);
    input_arg=rgb_data;     #input for the algorithm
    num_actions=5
    last_obs=rgb_data


    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
           
            return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    # Initialize target q function and q function
    Q = q_func(1, num_actions).type(dtype)
    target_Q = q_func(1, num_actions).type(dtype)

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(1000, 1)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    for t in count():


        last_idx = replay_buffer.store_frame(last_obs)

        recent_observations = replay_buffer.encode_recent_observation()

        # Choose random action if not yet start learning
        if t > learning_starts:
            action = select_epilson_greedy_action(Q, recent_observations, t)[0, 0]
        else:
            action = random.randrange(num_actions)
        # Advance one step

        control_robot(action + 1)

        rgb_data=depth_data.reshape(640,480,1)
        obs=rgb_data
        ##evaluate the action
        dis_data=np.array(depth_data)
        dis_data[np.isnan(dis_data)]=999999999999
        dis_data[dis_data==0]=999999999999
        dis=np.min(dis_data)
        print("MIN DISTANCE:"+str(dis)+"-------------")
        reward = 0
        if dis<500:
            reward = 1
        else:
            reward =-1
        print("REWARD:"+str(reward)+"--------------")
        # clip rewards between -1 and 1
        reward = max(-1.0, min(reward, 1.0))
        # Store other info in replay memory
        replay_buffer.store_effect(last_idx, action, reward, False)
        # Resets the environment when reaching an episode boundary.
        #if done:
            #obs = env.reset()
        last_obs = obs


        if (t > 1 and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            print("Training")
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
            act_batch = Variable(torch.from_numpy(act_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()

            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # We choose Q based on action taken.
            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze()
            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            next_max_q = target_Q(next_obs_batch).detach().max(1)[0]
            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            target_Q_values = rew_batch + (gamma * next_Q_values)
            print("next:",next_Q_values.shape)
            print("current:",current_Q_values.squeeze().shape)
            # Compute Bellman error
            bellman_error = target_Q_values - current_Q_values
            # clip the bellman error between [-1 , 1]
            clipped_bellman_error = bellman_error#.clamp(-1, 1)
            #print(clipped_bellman_error)
            # Note: clipped_bellman_delta * -1 will be right gradient
            d_error = clipped_bellman_error * -1.0
            # Clear previous gradients before backward pass
            #print(d_error.data)
            optimizer.zero_grad()
            # run backward pass
            current_Q_values.backward(d_error.data)

            # Perfom the update
            optimizer.step()
            num_param_updates += 1

            # Periodically update the target network by Q network to target Q network
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())


