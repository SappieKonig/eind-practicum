# process division into split frames
import glob
import cv2
import numpy as np
import utils
import torch
import os
import shutil
import matplotlib.pyplot as plt

split_dir = '/home/sappie/PycharmProjects/DoublePendulum/eind-practicum/camera_project_footage_edited'
videos = glob.glob(split_dir + '/*v2.mp4')
print(videos)

device = 'cuda'

dr_max = 30

def process_video(frames, stride):
    frames = torch.from_numpy(frames[..., 0]).to(device).float()
    frames = utils.extract_position_pipeline(frames, stride)
    return frames

def sub_split_video(video_file_name, parts: int=100):
    imgs = []
    new_vid = []
    vid = cv2.VideoCapture(video_file_name)
    success, image = vid.read()
    x, y = np.zeros(0), np.zeros(0)
    stride = 3

    size = 10

    # out = cv2.VideoWriter('/home/sappie/PycharmProjects/DoublePendulum/eind-practicum/test.mp4',
          # cv2.VideoWriter_fourcc(*'mp4v'), 60, (image.shape[1], image.shape[0]), True)

    while success:
        imgs.append(image)
        success, image = vid.read()
        if len(imgs) >= parts:
            frames = np.stack(imgs, axis=0)
            add_y, add_x = process_video(frames, stride)
            y, x = np.concatenate([y, add_y.cpu().numpy()]), np.concatenate([x, add_x.cpu().numpy()])
            # for i, frame in enumerate(frames):
            #     x_min = int(max(0, add_x[i] - size)) * stride
            #     x_max = int(min(frame.shape[1], add_x[i] + size)) * stride
            #     y_min = int(max(0, add_y[i] - size)) * stride
            #     y_max = int(min(frame.shape[0], add_y[i] + size)) * stride
            #     frame[y_min: y_max, x_min: x_max] = np.array([0, 0, 255])
            #     frame[x_min: x_max, y_min: y_max] = np.array([0, 0, 255])
            #     frame[x_min: x_max]
                # out.write(frame)

            # plt.plot(x, y)
            # plt.show()
            imgs = []

    # out.release()

    return x, y

for i, video in enumerate(videos):
    print(i)
    try:
        print(video)
        x, y = sub_split_video(video)
        coords = np.stack([x, y], axis=0)
        coords_file = video.replace('camera_project_footage_edited', 'coords').replace('.mp4', '.npy')
        np.save(coords_file, coords)
    except Exception as E:
        print(E)