import cv2


def load_offside_image(image_path:str):
    '''
    loads the image BGR components for a possible offside image

    parameters:
        image_path(str): path to the image containing a possible offside play

    returns:
        image BGR components
    '''
    offside_image = cv2.imread(image_path)
    return offside_image


def convert_to_HSV_color_space(image):
    '''
    converts the images BGR components to the HSV color space

    parameters:
        image(numpy array): array containing the BGR components

    returns:
        image(numpy array): array containing the HSV components
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image


