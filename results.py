"""Project 2: results script
We used a picture we took of kitchen objects, they can be found under test_data
The script uses right_4.jpn and left_4.jpg
"""

import cv2
import stereo
import numpy


if __name__ == '__main__':
    # Read initial images
    img_right = cv2.imread("test_data/right_4.jpg")
    img_left = cv2.imread("test_data/left_4.jpg")

    F, H_left, H_right = stereo.rectify_pair(img_left, img_right)

    rectified_left = stereo.warp_image(img_left, H_left)
    cv2.imwrite("test_data/rect_left.jpg", rectified_left)

    rectified_right = stereo.warp_image(img_right, H_right)
    cv2.imwrite("test_data/rect_right.jpg", rectified_right)

    disparity = stereo.disparity_map(rectified_left, rectified_right)
    cv2.imwrite("test_data/disparity.jpg", disparity)
    disparity_image = cv2.imread('test_data/disparity.png',
                                 cv2.CV_LOAD_IMAGE_GRAYSCALE)

    colors = cv2.imread('test_data/left_4.jpg')
    focal_length = 10

    ply_string = stereo.point_cloud(disparity, colors, focal_length)
    # View me in Meshlab!
    with open("our_results.ply", 'w') as f:
        f.write(ply_string)
