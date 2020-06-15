import argparse
import os

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

import time

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    print(height, net_input_height_size, net_input_height_size / height)
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    print(scaled_img.shape)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    start = time.time()
    stages_output = net(tensor_img)
    end = time.time()
    print("Time taken", end-start)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, output_dir, height_size, cpu, track, smooth, generate_video):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    if generate_video:
        video_output_path = os.path.join(output_dir, output_dir.split("/")[-1] +"-annotations" + ".mp4")
        #(1080, 1920, 3)
        video_out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('m','p','4','v'), 10, (1920,1080))
    
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 33
    
    for idx, img in enumerate(image_provider):
        
        orig_img = img.copy()
        frame_file = os.path.join(output_dir, output_dir.split("/")[-1]+f"_{idx:06}.txt")

        with open(frame_file, "w") as frame_f:
            print("Input the model is", orig_img.shape)
            #print(output_path)
            heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(num_keypoints):  # 19th for bg
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
            current_poses = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = Pose(pose_keypoints, pose_entries[n][18])
                current_poses.append(pose)

            if track:
                track_poses(previous_poses, current_poses, smooth=smooth)
                previous_poses = current_poses

            annotations_hand = []
            annotations_head = []
            for pose in current_poses:
                pose.draw(img)
                print(pose.hands_bbox, pose.head_bbox)
                
                for hand_bbox in pose.hands_bbox:
                    x_center = (hand_bbox[0][0] + hand_bbox[1][0]) / (2*1920)
                    y_center = (hand_bbox[0][1] + hand_bbox[1][1]) / (2*1080)
                    x_offset = (hand_bbox[1][0] - hand_bbox[0][0]) / 1920
                    y_offset = (hand_bbox[1][1] - hand_bbox[0][1]) / 1080
                    if x_center < 0.0:
                        x_offset = x_offset - x_center
                        x_center = 0
                    hand_bbox_scaled = (x_center, y_center, x_offset, y_offset)
                    print(hand_bbox_scaled)
                    frame_f.write("8 " + ' '.join(map(str, hand_bbox_scaled)) + "\n")
                    annotations_hand.append(hand_bbox_scaled)

                x_center = (pose.head_bbox[0][0] + pose.head_bbox[1][0]) / (2*1920)
                y_center = (pose.head_bbox[0][1] + pose.head_bbox[1][1]) / (2*1080)
                x_offset = (pose.head_bbox[1][0] - pose.head_bbox[0][0]) / 1920
                y_offset = (pose.head_bbox[1][1] - pose.head_bbox[0][1]) / 1080
                if x_center < 0.0:
                        x_offset = x_offset - x_center
                        x_center = 0
                head_bbox_scaled = (x_center, y_center, x_offset, y_offset)
                print(head_bbox_scaled)
                frame_f.write("9 " + ' '.join(map(str, head_bbox_scaled)) + "\n")
                annotations_head.append(head_bbox_scaled)

            img = cv2.addWeighted(orig_img, 0.3, img, 0.7, 0)
            for pose in current_poses:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                if track:
                    cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
            img_filename = os.path.join(output_dir, output_dir.split("/")[-1] + f"_{idx:06}.jpg")
            print(img_filename)
            cv2.imwrite(img_filename, orig_img)
            if generate_video:
                video_out.write(img)


            key = cv2.waitKey(delay)
            if key == 27:  # esc
                return
            elif key == 112:  # 'p'
                if delay == 33:
                    delay = 0
                else:
                    delay = 33
        if video_out:
            video_out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--video_dir', type=str, required=False)
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    parser.add_argument('--generate_video', type=bool, default=True)

    args = parser.parse_args()

    if args.video == '' and args.images == '' and args.video_dir == '':
        raise ValueError('Either --video or --image or --video_dir has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    if args.images or args.video:
        if args.images != "":
            frame_provider = ImageReader(args.images)
            output_dir = os.path.join(args.output_dir, args.images.split("/")[-1] + "_hand_head_annotations")
            #os.mkdir(output_dir) if (not os.isdir(output_dir))
            os.makedirs(output_dir, exist_ok=True)
            args.generate_video = False
        elif args.video != "":
            frame_provider = VideoReader(args.video)
            output_dir = os.path.join(args.output_dir, args.video.split("/")[-1].split(".")[0] + "_hand_head_annotations")
            os.makedirs(output_dir, exist_ok=True)
        run_demo(net, frame_provider, output_dir, args.height_size, args.cpu, args.track, args.smooth, args.generate_video)

    if args.video_dir:
        for file in os.listdir(args.video_dir):
            if file.find("1fps"):
                print(file)
                video_tag = file.split(".")[0]
                # create output folder
                output_dir = os.path.join(args.output_dir, video_tag + "_hand_head_annotations")
                os.makedirs(output_dir, exist_ok=True)
                frame_provider = VideoReader(os.path.join(args.video_dir, file))
                run_demo(net, frame_provider, output_dir, args.height_size, args.cpu, False, args.smooth, args.generate_video)
