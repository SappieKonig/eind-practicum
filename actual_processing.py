# process division into split frames
import glob
import cv2
import numpy as np
import utils
import torch
import os
import shutil
import matplotlib.pyplot as plt

split_dir = '/home/sappie/PycharmProjects/DoublePendulum/eind-practicum/camera_project_footage'
videos = glob.glob(split_dir + '/*.mp4')

colors = {'green': [.3, .6, .5], 'purple': [.6, .5, .85]}

device = 'cuda'

def save(frames, dir, name):
    dump_dir = os.path.join(dir, 'dump')
    try:
        os.mkdir(dump_dir)
    except FileExistsError:
        pass

    for i, frame in enumerate(frames):
        frame = frame[0].cpu().numpy()
        cv2.imwrite(f'{dump_dir}/{i}.jpg', np.stack([frame, frame, frame], axis=2))
    os.system(f"ffmpeg -r 60 -start_number 0 -i {dump_dir}/%d.jpg"
              f" -vcodec mpeg4 -y {dir}/{name}_processed.mp4")
    shutil.rmtree(dump_dir)



def process_video(frames, color):
    frames = torch.from_numpy(frames).to(device).float()
    frames = utils.accentuation_pipeline(frames, color)
    return frames

def sub_split_video(video_file_name, parts: int=100, color='green'):
    imgs = []
    new_vid = []
    vid = cv2.VideoCapture(video_file_name)
    success, image = vid.read()
    # print(video_file_name, success, image)
    # print(image.shape, 'shape')
    new_file_name = video_file_name[:-4] + '_' + color + '.mp4'
    new_file_name = new_file_name.replace('camera_project_footage', 'camera_project_footage_edited')
    # print(new_file_name)
    out = cv2.VideoWriter(new_file_name,
                          cv2.VideoWriter_fourcc(*'mp4v'), 60, (image.shape[1], image.shape[0]), True)
    while success:
        imgs.append(image)
        success, image = vid.read()
        if len(imgs) >= parts:
            frames = np.stack(imgs, axis=0)
            vid_part = process_video(frames, colors[color][::-1])
            for frame in vid_part:
                out.write(np.stack([frame.cpu().numpy()[0] for _ in range(3)], axis=-1))
            new_vid += [None]
            imgs = []
            # print('sth', len(new_vid))

        # if len(new_vid) == 50:
        #     break

    out.release()
    print(f'video length: {parts * len(new_vid)}')

    # vid_id = video_file_name.split('/')[-1].split('.')[0]

    # save([frame for sublist in new_vid for frame in sublist],
    #      os.path.dirname(video_file_name), vid_id)


for i, video in enumerate(videos):
    print(f'{i / len(videos) * 100:.2f}% complete', video)
    for color in colors:
        print(color)
        try:
            sub_split_video(video, color=color)
        except AttributeError as E:
            print(E)