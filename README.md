# Simulating Unsupervised Image Classification in Label-Scarce Environments

In this Jupyter notebook, I attempt to perform MNIST image classification by minimizing the number of calls to labels for images, in order to simulate working an an environment that is scarce of labeled data.

First, I use a deep auto-encoder architecture to compress the 768-dimensional representation of MNIST images to 32 dimensions and to extract features:

<p align="center">
  <img height="330" width="450" src="https://raw.githubusercontent.com/arvindrajaraman/label-scarce-img-classification/master/autoencoder_arch.png?token=AHGXVO3UTPVIGVUN3GJV3HLA3VDEM">
</p>

Then, I iteratively perform k-means clustering. In a certain iteration where we divide the data into k clusters, I find the medoid of each cluster. The medoid of a cluster is the closest datapoint in the dataset to the centroid ("average" or numerical center) of that cluster. I query the medoid's label and assign that label to the entire cluster. I specifically query for the medoid's label in a cluster, because datapoints closer to the fringes have a higher chance of being misclassified; the medoid's label is more likely to be representative of the cluster.

If for a certain k, the desired accuracy is not reached, then k is doubled and we run k-means clustering again. The reason why we geometrically increase k (doubling) instead of arithmetically increasing k (adding a fixed value to k in each iteration), because as follows: Let us say that the optimal k where the accuracy is just above or equal to 0.9 is k'. Then, the total images queried is as follows:

1 + 2 + 4 + ... + k' = 2k' - 1.

In other words, the total images queried will be approximately twice compared to if we knew that k' was the optimal k before-hand (and we could just set k = k' and query for k' images from the start). This is a good bound.

In total, I queried for 310 images' labels. This means that (60000-310)/(60000) = 99.5% of the dataset doesn't need to be labeled.
