# Segmentation Submission for aUToronto
This is my attempt at solving the aUToronto Perception Detection - 2D Object Detection Team's coding challenge. I did not have a lot of experience with image segmentation but wanted to see how much I could learn in two-three days. 

## How my code works
This program uses k-means clustering algorithm to partition the input image into clusters. An attempt is made to optimize the cluster number by using the [Elbow Method](https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/#:~:text=Elbow%20Method,-In%20the%20Elbow&text=WCSS%20is%20the%20sum%20of,is%20largest%20when%20K%20%3D%201.). To detect the region where the rate of change of inertia starts to decrease, a for loop compares the change in intertia between three consecutive points to detect the "elbow". Based on this the number of clusters is detected and used to perform k-means clustering. TO view the elbow curve and compare it to the number of clusters chosen uncomment the following block of code in the file: 

''' 
    ##uncomment this section to view the results of the elbow method and compare it with the chosen k value
    #plt.plot(K, inertia,'bx-') 
    #plt.title('Elbow Method')
    #plt.xlabel('Values of K') 
    #plt.ylabel('Inertia') 
    #plt.show()
'''

After segmenting the image every cluster but one is masked to black. After testing the clusters I saw that the bin cluster was consistently the 3rd one. This method of masking is not ideal, and further discussed below.  

## How to run my code
Required Libraries: numpy, matplotlib, cv2, scikit-learn
Additional Libraries: tkinter, os
Run segment.py and choose an input image when prompted to do so. The output image is saved under the output folder. 

## Issues/Next Steps 
1. Optimizing cluster count 
The current method of finding the "elbow" of the curve for cluster count needs improvement. 

2. Choosing clusters to mask 
The current code assumes that the optimal number of clusters will always be around 3-5, and based on that chooses the third cluster to keep. This could easily backfire for new images. I tried to intelligently detect the cluster by also segmenting by color (as the bins are always red) and looking matching the two clusters to find the "bin cluster" to save. I was not able to make this work, but pursuing this method will be my next step. I also considered using canny edge detection, but the edges of the bins tended to be well-recognized in the clusters anyways and I wasn't sure how to combine the edge findings with the cluster findings. That could also be a next step. 

3. Run-time 
The current run-time is not ideal, as it takes several minutes to segment one image. This is mostly due to the optimal_cluster() method. A next step would be to find a more efficient way for finding the cluster count. 