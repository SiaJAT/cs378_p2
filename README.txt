Write-up: Stereo
================
   
To run the test script, run the following commands in the terminal:
 
  * python results.py
 
This script will write the following images to the "test_data" directory

  1. "rect_left.jpg" - our left rectified image
  2. "rect_right.jpg" - our right rectified image
  3. "disparity.jpg" - a disparity image generated from our left and right rectified image
  4. "our_results.ply" - a ply file representing a point cloud our disparity

Also in our "test_data" directory is are images of apoint cloud generated from an image in the
Middlebury Stereo Datasets.  These images are "middlebury_1.jpg" and "middleburry_2.jpg".

We included the point-clouds for our results and the tsukuba as gzip files.
To unzip the point-clould file of intrest type gunzip <filename>