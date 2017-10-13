import cv2
import sys


def extract_frames(video_path):
    capture = cv2.VideoCapture(video_path)
    ret_val, frame = capture.read()
    frames = []
    idx = 0

    while ret_val is True and idx < 1000:
        print('extract frame {}'.format(idx))
        save_name = '{:05d}.png'.format(idx)
        cv2.imwrite(save_name, frame)
        # save out the frame

        idx += 1
        frames.append(frame)
        ret_val, frame = capture.read()


if __name__ == '__main__':
    video = sys.argv[1]
    extract_frames(video)
