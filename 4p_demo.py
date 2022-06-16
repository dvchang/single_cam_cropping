import tensorflow as tf
import cv2
import time
import argparse
import numpy as np

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=3376)
parser.add_argument('--cam_height', type=int, default=2528)
parser.add_argument('--scale_factor', type=float, default=0.3)
parser.add_argument('--file', type=str, default="/Users/dvc/pyproject/posenet-python/video/819PGR01P0803405_VID_20220607_162808.mp4",
                    help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def gen_crop(x1, y1, x2, y2):
    return [x1, y1, x2, y2]


def read_from_crop(crop1):
    return crop1[0], crop1[1], crop1[2], crop1[3]    
    
def area(a, b):  # returns None if rectangles don't intersect
    axmin, aymin, axmax, aymax = read_from_crop(a)
    bxmin, bymin, bxmax, bymax = read_from_crop(b)

    dx = min(axmax, bxmax) - max(axmin, bxmin)
    dy = min(aymax, bymax) - max(aymin, bymin)
    if (dx >= 0) and (dy >= 0):
        return dx*dy


def overlapping_perct(crop1, crop2):
    x1, y1, x2, y2 = read_from_crop(crop1)
    area1 = (x2 - x1) * (y2 - y1)
    overlap = area(crop1, crop2)
    return area1 / overlap


def crop_rect(nose, w, h, adjust, old_crop):
    
    yy = round(nose[1])
    xx = round(nose[0])

    y1 = yy - h + adjust
    y2 = yy + h + adjust
    x1 = xx - w
    x2 = xx + w
    new_crop = gen_crop(x1, y1, x2, y2)
    if (len(old_crop) <= 1):
        return new_crop
    over_lap = overlapping_perct(new_crop, old_crop)
    print('overlapping percentage :', over_lap)
    if (over_lap < 0.8):
        return new_crop
    return old_crop

def crop_image(img, crop):
    x1, y1, x2, y2 = read_from_crop(crop)
    cropped_image = img[y1:y2, x1:x2]
    return cropped_image

def generate_crop(img, nose, w, h, adjust, old_crop):
    crop_r = crop_rect(nose, w, h, adjust, old_crop)
    return crop_image(img, crop_r), crop_r

def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        result = cv2.VideoWriter('raw3k_out.avi',
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 30, (args.cam_width, args.cam_height))
        v_width_h = 300
        v_height_h = 400
        v_height_adjustment = 100
        compose = cv2.VideoWriter('compose_out.avi',
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  30, (v_width_h * 4 + 4, v_height_h*2))

        start = time.time()
        frame_count = 0
        crop1 = []
        crop2 = []
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            #cv2.imshow('posenet', overlay_image)

            pose = posenet.get_pose(
                pose_scores, keypoint_scores,
                keypoint_coords,
                min_pose_score=0.15,
                min_part_score=0.1)
    
            pose.sort(reverse=True, key=posenet.pose_get_nose_x)
    
    
            if (len(pose) == 2):
                pose1 = pose[1]
                c1 = pose1[1]
                pose2 = pose[0]
                c2 = pose2[1]
                p1 = c1
                p2 = c2

                
                img1, crop1 = generate_crop(display_image, p1, v_width_h, v_height_h, v_height_adjustment, crop1)
                

                img2, crop2 = generate_crop(display_image, p2, v_width_h, v_height_h, v_height_adjustment, crop2)
                
                black_separator = np.zeros((800, 4, 3), np.uint8)
            
                
                horizontal = np.concatenate((img1, black_separator, img2), axis = 1)
                
                cv2.imshow('posenet', horizontal)
                
                compose.write(horizontal)
        
            
            #result.write(overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))
        result.release()
        compose.release()


if __name__ == "__main__":
    main()
