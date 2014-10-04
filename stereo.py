"""Project 2: Stereo vision.

In this project, you'll extract dense 3D information from stereo image pairs.
"""

import cv2
import math
import numpy


def rectify_pair(image_left, image_right, viz=False):
    """Computes the pair's fundamental matrix and rectifying homographies.

    Arguments:
      image_left, image_right: 3-channel images making up a stereo pair.

    Returns:
      F: the fundamental matrix relating epipolar geometry between the pair.
      H_left, H_right: homographies that warp the left and right image so
        their epipolar lines are corresponding rows.
    """

    sift = cv2.SIFT()

    kpA, descriptorsA = sift.detectAndCompute(image_left, None)
    kpB, descriptorsB = sift.detectAndCompute(image_right, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptorsA, descriptorsB, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.73 * n.distance:
            good.append(m)

    srcP = numpy.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstP = numpy.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    height, width, _ = image_left.shape;

    fundamental, mask = cv2.findFundamentalMat(srcP, dstP, method = cv2.cv.CV_FM_RANSAC, param1 = 1.0, param2 = 0.99)
    _, H_left, H_right = cv2.stereoRectifyUncalibrated(srcP, dstP, fundamental, (height, width), threshold = 10.0)

    return fundamental, H_left, H_right

def disparity_map(image_left, image_right):
    """Compute the disparity images for image_left and image_right.

    Arguments:
      image_left, image_right: rectified stereo image pair.

    Returns:
      an single-channel image containing disparities in pixels,
        with respect to image_left's input pixels.
    """

    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM(minDisparity = min_disp,
        numDisparities = num_disp,
        SADWindowSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 1,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        fullDP = False
    )

    disp = stereo.compute(image_left, image_right)
    disp = (disp-min_disp)/num_disp
    print "num channels"
    print len(disp.shape)

    # cv2.imshow('disparity', (disp-min_disp)/num_disp)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return disp.astype(numpy.uint8)

def point_cloud(disparity_image, image_left, focal_length):
    """Create a point cloud from a disparity image and a focal length.

    Arguments:
      disparity_image: disparities in pixels.
      image_left: BGR-format left stereo image, to color the points.
      focal_length: the focal length of the stereo camera, in pixels.

    Returns:
      A string containing a PLY point cloud of the 3D locations of the
        pixels, with colors sampled from left_image. You may filter low-
        disparity pixels or noise pixels if you choose.
    """
    pass
