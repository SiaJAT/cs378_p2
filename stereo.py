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

    kp_left, descriptors_left = sift.detectAndCompute(image_left, None)
    kp_right, descriptors_right = sift.detectAndCompute(image_right, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.73 * n.distance:
            good.append(m)

    srcP = numpy.float32([kp_left[m.queryIdx].pt
                         for m in good]).reshape(-1, 1, 2)
    dstP = numpy.float32([kp_right[m.trainIdx].pt
                         for m in good]).reshape(-1, 1, 2)
    height, width, _ = image_left.shape

    fundamental, mask = cv2.findFundamentalMat(
        srcP, dstP, method=cv2.cv.CV_FM_RANSAC, param1=1.0, param2=0.99)

    _, H_left, H_right = cv2.stereoRectifyUncalibrated(
        srcP, dstP, fundamental, (height, width), threshold=5.0)

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
    min_disp = 64
    num_disp = 112 - min_disp

    # set of parameters that pass the unit test
    stereo = cv2.StereoSGBM(minDisparity=min_disp,
                            numDisparities=num_disp,
                            SADWindowSize=window_size,
                            uniquenessRatio=10,
                            speckleWindowSize=100,
                            speckleRange=32,
                            disp12MaxDiff=1,
                            P1=8 * 3 * window_size ** 2,
                            P2=32 * 3 * window_size ** 2,
                            fullDP=False)

    disp = stereo.compute(image_left, image_right)
    disp = (disp - min_disp) / num_disp

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
    height, width, _ = image_left.shape

    # corrected the Q matrix to properly reproject image
    Q = numpy.float32([[1, 0, 0, (width / 2)],
                       [0, -1, 0, (height / 2)],
                       [0, 0, focal_length, 0],
                       [0, 0, 0, 1]])

    points = cv2.reprojectImageTo3D(disparity_image, Q)
    colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)

    mask = disparity_image > disparity_image.min()
    out_points = points[mask]
    out_colors = colors[mask]
    verts = numpy.hstack([out_points, out_colors])

    output = """ply
             format ascii 1.0
             element vertex %d
             property float x
             property float y
             property float z
             property uchar red
             property uchar green
             property uchar blue
             end_header\n""" % len(verts)

    # To center image point cloud in meshlab
    vert_offset = 500
    for row in verts:
        output += '%f %f %f %d %d %d\n' % \
            (row[0], row[1]+vert_offset, row[2], row[3], row[4], row[5])

    return output


# Used warp_image function from project 1 to produce rectified images
def warp_image(image, homography):
    """Warps 'image' by 'homography'

    Arguments:
      image: a 3-channel image to be warped.
      homography: a 3x3 perspective projection matrix mapping points
                  in the frame of 'image' to a target frame.

    Returns:
      - a new 4-channel image containing the warped input, resized to contain
        the new image's bounds. Translation is offset so the image fits exactly
        within the bounds of the image. The fourth channel is an alpha channel
        which is zero anywhere that the warped input image does not map in the
        output, i.e. empty pixels.
      - an (x, y) tuple containing location of the warped image's upper-left
        corner in the target space of 'homography', which accounts for any
        offset translation component of the homography.
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    height, width, depth = img.shape
    warp = cv2.warpPerspective(img, homography,
                               (width, height))
    return warp
