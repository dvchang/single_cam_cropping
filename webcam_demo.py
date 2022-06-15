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
parser.add_argument('--file', type=str, default="/Users/dvc/pyproject/posenet-python/video/raw3k.mp4",
                    help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def crop_image(img, nose, w, h):

    yy = round(nose[1])
    xx = round(nose[0])

    y1 = yy - h + 100
    y2 = yy + h + 100
    x1 = xx - w
    x2 = xx + w

    cropped_image = img[y1:y2, x1:x2]
    return cropped_image


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
        compose = cv2.VideoWriter('compose_out.avi',
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  30, (v_width_h * 4, v_height_h*2))

        start = time.time()
        frame_count = 0
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
    
            #pose.sort(key=posenet.pose_get_nose_x)
    
    
            if (len(pose) == 2):
                pose1 = pose[1]
                c1 = pose1[1]
                pose2 = pose[0]
                c2 = pose2[1]
                if (c1[0] <= c2[0]):
                    p1 = c1
                    p2 = c2
                else:
                    p1 = c2
                    p2 = c1
            
            
                
        
                
                img1 = crop_image(display_image, p1, v_width_h, v_height_h)
                

                img2 = crop_image(display_image, p2, v_width_h, v_height_h)
                
                horizontal = np.concatenate((img1, img2), axis = 1)
                
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
