import cv2
import sys
import numpy as np
from utilities import convert_to_HSV_color_space, load_offside_image


def nothing(x):
    pass


def detect_football_field(image):
    image_HSV = convert_to_HSV_color_space(image)
    cv2.namedWindow('threshold')
    cv2.createTrackbar('lowH','threshold',0,255,nothing)
    cv2.createTrackbar('highH','threshold',255,255,nothing)
    cv2.createTrackbar('lowS','threshold',0,255,nothing)
    cv2.createTrackbar('highS','threshold',255,255,nothing)
    cv2.createTrackbar('lowV','threshold',0,255,nothing)
    cv2.createTrackbar('highV','threshold',255,255,nothing)

    while(True):
        # get trackbar positions
        ilowH = cv2.getTrackbarPos('lowH', 'threshold')
        ihighH = cv2.getTrackbarPos('highH', 'threshold')
        ilowS = cv2.getTrackbarPos('lowS', 'threshold')
        ihighS = cv2.getTrackbarPos('highS', 'threshold')
        ilowV = cv2.getTrackbarPos('lowV', 'threshold')
        ihighV = cv2.getTrackbarPos('highV', 'threshold')

        lower_hsv = np.array([ilowH, ilowS, ilowV])
        higher_hsv = np.array([ihighH, ihighS, ihighV])
        mask = cv2.inRange(image_HSV, lower_hsv, higher_hsv)
        filtered_frame_BGR = cv2.bitwise_and(image, image, mask=mask)

        # show thresholded image
        cv2.imshow('image', filtered_frame_BGR)
        k = cv2.waitKey(1000) & 0xFF # large wait time to remove freezing
        if k == 113 or k == 27:
            break
    cv2.destroyAllWindows()


if __name__=="__main__":
    image_path = sys.argv[1]
    offside_image = load_offside_image(image_path)
    detect_football_field(offside_image)