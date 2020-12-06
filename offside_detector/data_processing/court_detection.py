import cv2
import sys
from utilities import convert_to_HSV_color_space, load_offside_image

def detect_football_field(image):
    image_HSV = convert_to_HSV_color_space(image)
    
    cv2.imshow('offside image', image_HSV)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    image_path = sys.argv[1]
    offside_image = load_offside_image(image_path)
    detect_football_field(offside_image)