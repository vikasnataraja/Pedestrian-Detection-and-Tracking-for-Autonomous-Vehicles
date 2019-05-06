# Pedestrian-Detection-and-Tracking-for-Autonomous-Cars
This project uses Histogram of Oriented Gradients for pedestrian detection and Kalman Filter for tracking and prediction

## What is this repo?

The main objective behind this project is to devise an algorithm to identify and track pedestrians from the eyes of a moving vehicle. This would enable the vehicle to know the scene around it (often called scene understanding in the industry) and make decisions. The second part of the goal, which is pedestrian tracking, is highly useful because the vehicle can then plan ahead to avoid those paths.

Ideally, I would have liked to develop the whole algorithm from scratch but owing to time constraints, I decided to stick to these 2 goals.


## What principles are being used here:

* HOG (Histogram of Oriented Gradients) for the detection part
* Kalman Filter for tracking and prediction


## What you will need to run this:

* Python 3 
* OpenCV (>3.0.0)
* Video file (for this I used a video called "london_bus.mp4" which is provided in the repo. You can use other videos but make sure you edit the code to change the name or directory)

## How to run:

* Navigate to "pedestrians.py"
* Comment out line 11 if you have OpenCV installed in the same directory as the other packages
* Run using any Python 3 interpreter

## References
1. Student Dave’s Tutorials, http://studentdavestutorials.weebly.com/multi-bugobject-tracking.html
2. HOG Person Detector, Chris McCormick, http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/ , 2013
3. Chen, Xi & Wang, Xiao & Xuan, Jianhua, VPISU, “Tracking Multiple Objects using Unscented Kalman Filtering Techniques”, https://arxiv.org/ftp/arxiv/papers/1802/1802.01235.pdf
4. KITTI, http://www.cvlibs.net/datasets/kitti/eval_tracking.php
5. Object Detection, OpenCV, https://docs.opencv.org/3.4/d5/d54/group__objdetect.html
