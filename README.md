## Vehicle Detection Project

The goals / steps of this project are the following:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a
labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color
features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and
randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search
for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later
implement on full project_video.mp4) and create a heat map of recurring
detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Data Exploration

* Number of vehicle images: 8792
* Number non-vehicle images: 8968

First, I explored different skimage.hog() parameters ( orientations , pixels_per_cell , and
cells_per_block ). I grabbed random images from each of the two classes and displayed
them to get a feel for what the skimage.hog() output looks like. 

Here is an example using the RGB color space and HOG parameters of orientations=9 ,
pixels_per_cell=(8, 8) and cells_per_block=(2, 2) :

I also visualized the color histograms and the spatial bins (RGB):

I tried various combinations of parameters (code cell for all features on a sample of
1000 and then 2000 images from vehicles and non vehicles set. The best fit with over
99% accuracy was achieved with the following params:
* color_space='YCrCb'
* spatial_size=(32, 32)
* hist_bins=32
* orient=9
* pix_per_cell=8
* cell_per_block=2
* hog_channel='ALL’

### Classifier 

I trained a linear SVM with a feature vector that combined spatially binned features,
color histogram features and hog features (code cell 11) and got an accuracy of over
99% on the test set:
* 41.66 seconds to extract vehicle features...
* 40.87 seconds to extract non-vehicle features...
* Feature vector length: 8460
* 22.3 seconds to train SVC..
* Test accuracy of SVC: 0.9904

### Sliding Window Search

With overlap = 0.5 and xy_window as (64, 64) i got the following results on the test
images:

It searched over 800 windows (some needlessly) and missed the white cars in several
of these images and finds cars in tree. So next I edited the y crop to not include the tree
and sky portion and set xy_window to 128. The results:

There are several false positives and it misses the cars that are far away (smaller
scale)

Single scale (1.5) window search without thresholding and without using
scipy.ndimage.measurements.label() to identify individual blobs in heat maps yielded
these results:

Ultimately I searched on multi scales (1 to 3.5) using YCrCb 3-channel HOG features
plus spatially binned color and histograms of color in the feature vector, which provided
a nice result. I also added heat map calculations and thresholding the heat map for
better results (cell 14). Here are some example images:

### Final Result

Here's a link to [my video result](https://github.com/shubhra/vehicle-detection/blob/master/output_video.mp4)

### False Positives & Overlapping Bounding Boxes

I recorded the positions of positive detections in each frame of the video. From the
positive detections I created a heatmap and then thresholded that map to identify
vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual
blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I
constructed bounding boxes to cover the area of each blob detected.

### Failure Points & Enhancements

Using multi scale window search is a little slow and could be made faster by averaging
over a few frames the position of a vehicle. But calculation over multiple scales does
rule out almost all false positives. Another possible speedup is to search using smaller
scale values near the horizon (just below) y values and larger as we get closer to the
car.
