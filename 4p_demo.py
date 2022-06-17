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
    return overlap / area1

def calculate_crop(crop, crop_target):
    x1, y1, x2, y2 = read_from_crop(crop)
    xx1, yy1, xx2, yy2 = read_from_crop(crop_target)
    m = 2
    mx = abs(x1 - xx1)
    if (mx > m):
        mx = m
    if (x1 > xx1):
        x1 = x1 - mx;
        x2 = x2 - mx;
    else:
        if (x1 < xx1):
            x1 = x1 + mx;
            x2 = x2 + mx;
    
    my = abs(y1 - yy1)
    if (my > m):
        my = m
    if (y1 > yy1):
            y1 = y1 - m;
            y2 = y2 - m;
    else:
        if (y1 < yy1):
            y1 = y1 + m;
            y2 = y2 + m;
    return gen_crop(x1, y1, x2, y2)


def calculate_crop_target(nose1, nose2, w, h, adjust, old_crop_target):
    ny1 = nose1[1]
    nx1 = nose1[0]
    ny2 = nose2[1]
    nx2 = nose2[0]
    
    yy = round((ny1 + ny2)/2)
    xx = round((nx1 + nx2)/2)

    y1 = yy - h + adjust
    y2 = yy + h + adjust
    x1 = xx - w
    x2 = xx + w
    new_crop = gen_crop(x1, y1, x2, y2)
    if (len(old_crop_target) <= 1):
        return new_crop
    over_lap = overlapping_perct(new_crop, old_crop_target)
    print('overlapping percentage :', over_lap)
    if (over_lap < 0.90):
        #print('overlapping percentage :', over_lap)
        return new_crop
    return old_crop_target

def crop_image(img, crop):
    x1, y1, x2, y2 = read_from_crop(crop)
    cropped_image = img[y1:y2, x1:x2]
    return cropped_image

def generate_crop(img, nose1, nose2, w, h, adjust, old_crop, old_crop_target):
    crop_target = calculate_crop_target(nose1, nose2, w, h, adjust, old_crop_target)
    if (len(old_crop) < 1):
        crop_r = crop_target
    else:
        crop_r = calculate_crop(old_crop, crop_target)
    return crop_image(img, crop_r), crop_r, crop_target

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
        v_width = 1280
        v_height = 720
        crop_width = round(v_width/4)
        factor = 1.3
        v_width_h = round((crop_width * factor) / 2)
        v_height_h = round((v_height * factor) / 2)
        v_height_adjustment = round(90 * factor)
        v_separator_w = 10
        compose = cv2.VideoWriter('compose_out.avi',
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  30, (v_width * 1 + v_separator_w * 3, v_height))

        start = time.time()
        frame_count = 0
        crop1 = []
        crop2 = []
        crop3 = []
        crop4 = []
        crop1_target = []
        crop2_target = []
        crop3_target = []
        crop4_target = []
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

            cv2.imshow('posenet', overlay_image)

            pose = posenet.get_pose(
                pose_scores, keypoint_scores,
                keypoint_coords,
                min_pose_score=0.15,
                min_part_score=0.1)
    
            pose.sort(key=posenet.pose_get_nose_x)
    
    
            if (len(pose) == 4):
                pose_a = pose[0]
                p1 = pose_a[1]
                pose_a = pose[1]
                p2 = pose_a[1]
                
                pose_a = pose[2]
                p3 = pose_a[1]
                pose_a = pose[3]
                p4 = pose_a[1]
                
                
                img1, crop1, crop1_target = generate_crop(display_image, p1, p1, v_width_h, v_height_h, v_height_adjustment, crop1, crop1_target)
                img1 = cv2.resize(img1, dsize=(crop_width, v_height), interpolation=cv2.INTER_CUBIC)
                
                
                img2, crop2, crop2_target = generate_crop(display_image, p2, p2, v_width_h, v_height_h, v_height_adjustment, crop2, crop2_target)
                img2 = cv2.resize(img2, dsize=(crop_width, v_height), interpolation=cv2.INTER_CUBIC)
 
                img3, crop3, crop3_target = generate_crop(display_image, p3, p3, v_width_h, v_height_h, v_height_adjustment, crop3, crop3_target)
                img3 = cv2.resize(img3, dsize=(crop_width, v_height), interpolation=cv2.INTER_CUBIC)
                
                img4, crop4, crop4_target = generate_crop(display_image, p4, p4, v_width_h, v_height_h, v_height_adjustment, crop4, crop4_target)
                img4 = cv2.resize(img4, dsize=(crop_width, v_height), interpolation=cv2.INTER_CUBIC)
                
                black_separator = np.zeros((v_height, v_separator_w, 3), np.uint8)
            
                
                horizontal = np.concatenate((img1, black_separator, img2, black_separator, img3, black_separator, img4), axis = 1)
                
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
