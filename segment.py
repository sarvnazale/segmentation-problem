##import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from tkinter import Tk
from tkinter.filedialog import askopenfilename 
import os


def optimal_cluster(dataset):
    ''' Uses sum of squared distance/elbow methd to find the intertia of eaah cluster. 
    Then compares the change in inertia to find the ideal number of clusters based on the graph.
    Returns k_value (cluster number) to be used in the kmeans function. 
        Parameters: 
            dataset: image in the form of pixel values 
    ''' 
    inertia = []
    K = range(1,10) #maximum of 10 clusters being considered 
    
    ##use sum of squared distance/elbow method to find inertia vs. k value 
    for num_clusters in K :
        kmeans = KMeans(n_clusters=num_clusters) 
        kmeans.fit(dataset)
        inertia.append(kmeans.inertia_) 

    ##attempt to intelligently discern k value from the elbow method
    for i in range(6): 
        k_value = 4 #default k value if condition fails 
        first_difference = inertia[i + 1] - inertia[i + 2]
        second_difference =  inertia[i + 2] - inertia[i +3]
        #if the difference between the three consecutive values is drastic, then pick the third value (past the "elbow" of the curve)
        if second_difference < 0.25 * first_difference: 
            k_value = i + 3 
            print("Chosen K value", k_value) #print chosen value to compare with elbow
            break

    ##uncomment this section to view the results of the elbow method and compare it with the chosen k value
    #plt.plot(K, inertia,'bx-') 
    #plt.title('Elbow Method')
    #plt.xlabel('Values of K') 
    #plt.ylabel('Inertia') 
    #plt.show()
    
    return k_value

def masking(k_value, labels, dataset, image, image_path): 
    ''' Masks all clusters but one by turning those clusters black. 
    Saves final masked image to output folder. 
        Parameters: 
            k_value: number of clusters
            labels: label array returned by cv2.kmeans
            dataset: image in the form of pixel values 
            image: original image to access its shape
            image_path: input path of original image '''
    masked_image = np.copy(dataset) 
    
    masked_image = masked_image.reshape((-1, 3)) # convert to the shape of pixel values
    
    #set all clusters but one to black
    for i in range(k_value): 
        if i == 3: 
            masked_image[labels == i] = [255, 255, 255] #set cluster to white
        else: 
            masked_image[labels == i] = [0, 0, 0]

    masked_image = masked_image.reshape(image.shape) #reshape to original shape
    plt.imshow(masked_image) #display new image
    output_path = "output/" + os.path.basename(image_path)
    print(output_path)
    cv2.imwrite(output_path, masked_image)

def main():
    ''' Saves image input path. 
    Processes image to prepare it for the cv2.kmeans method. 
    Calls the optimal_cluster() method to find the number of clusters and partition image. 
    Segments image. 
    Calls masking() method to mask segmented image. '''
    Tk().withdraw()
    image_path = askopenfilename()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image.reshape((-1, 3))

    pixel_values = np.float32(pixel_values) # convert to float as required by cv2.kmeans
    # setting stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = optimal_cluster(pixel_values) #detect optimal number of clusters
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers) #convert back to 8 bit values

    ## flatten the labels array
    labels = labels.flatten()
    ## convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    #plt.imshow(segmented_image, interpolation='none',cmap=plt.cm.jet,origin='upper' ) #show the clustered image as a colormap
    #plt.show()
    masking(k, labels, pixel_values, image, image_path) #mask undesired portions
    

if __name__ == "__main__":
    main()
