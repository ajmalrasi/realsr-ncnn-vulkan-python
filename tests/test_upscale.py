#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from tqdm import tqdm
import mimetypes
import logging
import glob
import os
from os import path as osp
from pathlib import Path
import ffmpeg
from PIL import Image, ImageChops, ImageStat
from realsr_ncnn_vulkan_python import RealSR, wrapped
import numpy as np
import cv2

# Use GPU 0 if available.
if wrapped.get_gpu_count() > 0:
    gpu_id = 2
    num_threads = 1
else:
    gpu_id = -1
    #  use all cores with CPU mode
    num_threads = os.cpu_count()


def _calc_image_diff(image0: Image.Image, image1: Image.Image) -> float:
    """
    calculate the percentage of differences between two images

    :param image0 Image.Image: the first frame
    :param image1 Image.Image: the second frame
    :rtype float: the percent difference between the two images
    """
    difference = ImageChops.difference(image0, image1)
    difference_stat = ImageStat.Stat(difference)
    percent_diff = sum(difference_stat.mean) / (len(difference_stat.mean) * 255) * 100
    return percent_diff

def get_video_meta_info(video_path):
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    return ret

def get_sub_video(args, num_process, process_idx):
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    os.makedirs(osp.join(args.output, f'{args.video_name}_inp_tmp_videos'), exist_ok=True)
    out_path = osp.join(args.output, f'{args.video_name}_inp_tmp_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin, f'-i {args.input}', '-ss', f'{part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '', '-async 1', out_path, '-y'
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path

class Reader:

    def __init__(self, args, total_workers=1, worker_idx=0):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='rgb24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])
            self.width, self.height = tmp_img.size
        self.idx = 0

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        rgb_image = cv2.cvtColor(np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3]), cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(rgb_image)
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        rgb_image = cv2.cvtColor(cv2.imread(self.paths[self.idx]))
        img = Image.fromarray(rgb_image)
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()
        else:
            return self.get_frame_from_list()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()


class Writer:

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec='hevc_nvenc',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec='hevc_nvenc',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        numpy_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        frame = numpy_frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

def run(args):

    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')
    img_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.jpg')

    reader = Reader(args)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    print("Height {}, width {}, FPS {}".format(height, width, fps))
    writer = Writer(args, audio, height, width, video_save_path, fps)
    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    upscaler = RealSR(gpu_id, num_threads=num_threads, scale=args.outscale)
    while True:
        img = reader.get_frame()
        if img is None:
            break
        
        output_image = upscaler.process(img)
        # output_image.save(img_save_path, 'JPEG')
        writer.write_frame(output_image)

        pbar.update(1)

    reader.close()
    writer.close()

def main():
    """Inference demo for Real-ESRGAN.
    It mainly for restoring anime videos.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')
    parser.add_argument('--num_process_per_gpu', type=int, default=1)
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    args = parser.parse_args()

    args.input = args.input.rstrip('/').rstrip('\\')
    os.makedirs(args.output, exist_ok=True)
    run(args)


if __name__ == '__main__':
    main()
