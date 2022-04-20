import numpy as np
import cv2 as cv
import os
import shutil
from skimage.util.shape import view_as_windows
import torch.nn.functional as F
import torch
from functools import lru_cache


@lru_cache(maxsize=None)
def get_exp_decay_filter(filter_size=1, decay=.9, avg=True):
    side = 2 * filter_size + 1
    filter = torch.ones((side, side))
    for i in range(side):
        for j in range(side):
            filter[i, j] *= decay ** (((filter_size - i) ** 2 + (filter_size - j) ** 2) ** 0.5)
    if avg:
        filter /= torch.sum(filter)
    return filter


def accentuate_color(distance):
    return -5*distance**2


def color_distance(frames, color, channel_weights=(1, 1, 1)):
    channel_weights = np.array(channel_weights)
    color = torch.from_numpy(np.reshape(color, (1, 1, 1, 3))).to(frames.device).float()
    color = color / torch.sum(color ** 2) ** .5
    frames = frames - color
    frames = frames ** 2
    frames = frames * torch.from_numpy(channel_weights).to(frames.device).view(1, 1, 1, 3).float()
    frames = torch.sum(frames, dim=-1) ** .5
    return frames


def normalize_colors(frame):
    return frame / torch.sum((frame+1) ** 2, dim=-1, keepdim=True) ** .5


def filter_image(frames, filter_size=1, decay=.9, avg=True):
    filter = get_exp_decay_filter(filter_size, decay, avg).to(frames.device)
    return F.conv2d(frames, filter.view(1, 1, filter.shape[0], filter.shape[1]),
                    padding=filter_size)


def accentuation_pipeline(frames, color, channel_weights=(1, 1, 1)):
    frames = normalize_colors(frames)
    frames = color_distance(frames, color, channel_weights)
    frames = accentuate_color(frames)

    # add channel dimension
    frames = torch.unsqueeze(frames, dim=1)
    for i in range(4):
        frames = filter_image(frames, decay=.9, avg=True)
    frames = frames - torch.amax(frames, dim=(1, 2, 3), keepdim=True)
    frames = torch.exp(frames * 5)
    frames = project_image_to_u8(frames)
    return frames


def localization_pipeline(img, color, channel_weights=(1, 1, 1)):
    return get_positions(accentuation_pipeline(img, color, channel_weights))


def save(frames, fps=30, dir='movie.mp4'):
    try:
        os.mkdir('dump')
    except FileExistsError:
        pass

    for i, frame in enumerate(frames):
        cv.imwrite(f'dump/img{i:0>5}.png', frame)
    os.system(f"ffmpeg -r {fps} -start_number 0 -i /home/sappie/PycharmProjects/DoublePendulum/dump/img%05d.png"
              f" -vcodec mpeg4 -y {dir}")
    shutil.rmtree('./dump')


def project_image_to_u8(frames):
    frames *= 255
    return frames.to(torch.uint8)


def get_positions(frame):
    return np.unravel_index(np.argmax(frame), frame.shape)


def extract_position_pipeline(frames, stride):
    frames = torch.unsqueeze(frames, dim=1)
    filter = torch.zeros((31, 31)).to(frames.device) - .3
    filter[5:26, 5:26] = get_exp_decay_filter(10, decay=.9, avg=False)
    filter /= torch.sum(torch.abs(filter))

    frames = F.conv2d(frames, filter.view(1, 1, filter.shape[0], filter.shape[1]),
                    padding=len(filter) // 2, stride=stride)


    positions = torch.argmax(frames.view((frames.shape[0], -1)), dim=1)

    y, x = positions // frames.shape[3], positions % frames.shape[3]
    return x, y
