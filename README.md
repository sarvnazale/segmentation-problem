# Segmentation Submission for aUToronto
This is my attempt at solving the aUToronto Perception Detection - 2D Object Detection Team's coding challenge. I did not have a lot of experience with image segmentation but wanted to see how much I could learn in two-three days. 

## How my code works
This program uses k-means clustering algorithm to partition the input image into clusters. An attempt is made to optimize the cluster number by using the [Elbow Method](https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/#:~:text=Elbow%20Method,-In%20the%20Elbow&text=WCSS%20is%20the%20sum%20of,is%20largest%20when%20K%20%3D%201.). To detect the region where the rate of change of inertia starts to decrease, a for loop compares the change in intertia between three consecutive points to detect the "elbow". Based on this the number of clusters is detected and used to perform k-means clustering. 

## How to run my code
Required Libraries: numpy, matplotlib, cv2, scikit-learn
Additional Libraries: tkinter, os
Run segment.py and choose an input image when prompted to do so. The output image is saved under the output folder. 

## Issues/Next Steps 
1. Optimizing cluster count 
The current method of finding the "elbow" of the curve for cluster count needs improvement. 

2. Choosing clusters to mask 
The current code assumes that the cluster code does 