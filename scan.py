import cv2
import numpy as np

##############################
img_height = 480
img_width = 640
t_lower = 200
t_higher = 200
kernel_h = 5
kernel_w = 5
sigma_X = 1
##############################


def preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(imgGray, (kernel_h, kernel_w), sigma_X)
    img_canny = cv2.Canny(img_blur, t_lower, t_higher)
    kernel = np.ones((kernel_h, kernel_w))
    # Double Dilation; To Increase the width of the edges
    img_dial = cv2.dilate(img_canny, kernel, iterations=2)
    # Shrinking the edges little bit, also removing noise
    img_eroded = cv2.erode(img_dial, kernel, iterations=1)
    return img_eroded


def get_contours(img):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(img_cont, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(img_cont, biggest, -1, (255, 0, 0), 20)
    return biggest


def get_warp(img, biggest):
    biggest = reorder(biggest)
    print(biggest.shape)
    pts1 = np.float32(biggest)
    pts2 = np.float32(
        [[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img, matrix, (img_width, img_height))
    cropped_img = img_output[20:img_output.shape[0] -
                             20, 20:img_output.shape[1]-20]
    cropped_img = cv2.resize(cropped_img, (img_width, img_height))
    return cropped_img


def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def stack_images(scale, images):
    rows = len(images)
    cols = len(images[0])
    rowsAvailable = isinstance(images[0], list)
    width = images[0][0].shape[1]
    height = images[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if images[x][y].shape[:2] == images[0][0].shape[:2]:
                    images[x][y] = cv2.resize(
                        images[x][y], (0, 0), None, scale, scale)
                else:
                    images[x][y] = cv2.resize(
                        images[x][y], (images[0][0].shape[1], images[0][0].shape[0]), None, scale, scale)
                if len(images[x][y].shape) == 2:
                    images[x][y] = cv2.cvtColor(
                        images[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(images[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if images[x].shape[:2] == images[0].shape[:2]:
                images[x] = cv2.resize(images[x], (0, 0), None, scale, scale)
            else:
                images[x] = cv2.resize(
                    images[x], (images[0].shape[1], images[0].shape[0]), None, scale, scale)
            if len(images[x].shape) == 2:
                images[x] = cv2.cvtColor(images[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(images)
        ver = hor
    return ver


image_path = "sample.png"
img = cv2.imread(filename=image_path)
# img.set(3, img_width)
# img.set(4, img_height)
# img.set(10, 130)

img = cv2.resize(img, (img_width, img_height))
img_cont = img.copy()
img_thres = preprocessing(img)
biggest = get_contours(img_thres)
img_warped = get_warp(img_cont, biggest)
image_array = ([img, img_cont], [img_thres, img_warped])
stacked_images = stack_images(0.6, image_array)
cv2.imshow('Text Image', stacked_images)
