'''

roi_x_start, roi_y_start, roi_x_end, roi_y_end = 0, 0, 0, 0
cropping = False
roi_selected = False
roi_reference_pts = []


def click_and_crop(event, x, y, flags, param):
    
    This is the Mouse callback function, which handles how the algorithm deals with the inputs from
    the mouse when moving on the defined window

    parameters:
        event: one of the cv::MouseEventTypes constants.
        x: x-axis mouse position when placed on the defined window
        y: y-axis mouse position when placed on the defined window
        flags: one of the cv::MouseEventFlags constants.

    
    # grab references to the global variables
    global roi_x_start, roi_y_start, roi_x_end, roi_y_end, cropping, roi_selected

    # if the left mouse button was clicked, record the starting (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_x_start, roi_y_start, roi_x_end, roi_y_end = x, y, x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            roi_x_end, roi_y_end = x, y

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that the cropping operation is finished
        roi_x_end, roi_y_end = x, y
        cropping = False
        roi_selected = True


def detect_football_field(image):

    # grab references to the global variables
    global roi_x_start, roi_y_start, roi_x_end, roi_y_end, cropping, roi_selected
    
    # load the image, clone it, and setup the mouse callback function
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    
    while True:
        i = image.copy()
        if not cropping and not roi_selected:
            cv2.imshow("image", image)

        elif cropping and not roi_selected:
            cv2.rectangle(i, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
            cv2.imshow("image", i)

        elif not cropping and roi_selected:
            cv2.rectangle(image, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
            cv2.imshow("image", image)

        key = cv2.waitKey(1) & 0xFF
    
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            roi_selected = False
    
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    roi_reference_pts = [(roi_x_start, roi_y_start), (roi_x_end, roi_y_end)]
    if len(roi_reference_pts) == 2:
        roi = clone[roi_reference_pts[0][1]:roi_reference_pts[1][1], roi_reference_pts[0][0]:roi_reference_pts[1][0]]
        #cv2.imshow("ROI", roi)

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        print('min H = {}, min S = {}, min V = {}; max H = {}, max S = {}, max V = {}'.format(hsv_roi[:,:,0].min(), hsv_roi[:,:,1].min(), hsv_roi[:,:,2].min(), hsv_roi[:,:,0].max(), hsv_roi[:,:,1].max(), hsv_roi[:,:,2].max()))
    
        lower_hsv = np.array([hsv_roi[:,:,0].min(),0,0]) #,hsv_roi[:,:,1].min(),hsv_roi[:,:,2].min()])
        higher_hsv = np.array([hsv_roi[:,:,0].max(),255,255]) #,hsv_roi[:,:,1].max(),hsv_roi[:,:,2].max()])
        mask = cv2.inRange(clone, lower_hsv, higher_hsv)
        filtered_frame_BGR = cv2.bitwise_and(clone, clone, mask=mask)
        
        #image_to_thresh = clone
        #hsv = cv2.cvtColor(image_to_thresh, cv2.COLOR_BGR2HSV)

        #kernel = np.ones((3,3),np.uint8)
        # for red color we need to masks.
        #mask = cv2.inRange(hsv, lower, upper)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('offside image', filtered_frame_BGR)

        cv2.waitKey(0)
    # close all open windows
    cv2.destroyAllWindows()

'''
'''
    filtered_frame_grey = cv2.cvtColor(filtered_frame_BGR,cv2.COLOR_BGR2GRAY)
    #Defining a kernel to do morphological operation in threshold #image to get better output.
    kernel = np.ones((13,13),np.uint8)
    thresh = cv2.threshold(filtered_frame_grey,127,255,cv2.THRESH_BINARY_INV |  cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('offside image', thresh)
    cv2.waitKey(0)
'''