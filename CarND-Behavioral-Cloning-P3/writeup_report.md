

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data/Udacity provided sample data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


---
### Files Submitted

#### 1. Submission includes all required files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* model.h5_first_track a model that fits the first track (version 1)
* model.h5_first_track_and_second_partial a model that fits the first track and fits the second one , but the last turn. Trained only using the first track data.
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and udacity provided sample data, and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

[//]: # (Image References)

[image1]: ./road.png "Road"
[image2]: ./road_clipped.png "Clipped Road"
[image3]: ./first.png "Steering angle Distribution"
[image4]: ./distribution_after.png "After augmentation"


#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
Table below describes the layers.

| Layers   |   Specification      | Remarks |
|----------|:-------------:|------:|
| Lambda  |  Normalization |   |
| Convolution2D |  3x3 filter, 32 channels   |    |
| Activation | relu |    |
| Maxpooling2D | pool_size = 2,2 |    |
| Convolution2D |  3x3 filter, 64 channels    |    |
| Activation | relu |    |
| Maxpooling2D | pool_size = 2,2  |    |
| Convolution2D |  3x3 filter,128 channels    |    |
| Activation | relu |    |
| Maxpooling2D | pool_size = 2,2 |     |
| Flatten |      |    |
| Dense | 128 |    |
| Dropout | probability = 0.6| |
| Activation | relu |    |
| Dense | 84 |    |
|Dropout | probability = 0.6| |
| Activation | relu | |
| Dense | 1 |  steering angle  |



#### 2. Attempts to reduce overfitting in the model

Dropout layer and Gradient Clipping (-1,1) was used. Also, did try out adding a small gaussian noise to the steering angle (When using augmentation , and oversampling, but did not yield better result).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Also, gradient clipping was used.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road , with a correction of 0.2.

Udacity provided data was used for training.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use Convolution layer and Pooling layers to detect the edges which are needed for lane detection. Followed by couple of fully connected layer , with dropout layers to avoid over fitting.

My first step was to use a convolution neural network model similar to the LeNet architecture of which this model is based.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set( 80 - 20 split) .

Multiple model with few conv-maxpool layers were tried , and found that the current architecture worked best for the first track.

#### 2. Final Model Architecture

The final model architecture is as explained in the table above.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the udacity provided data set, used the left and right camera images with a correction factor of 0.2. Also, augmented the data with flip images of all the type of images (left , center , right).
Also, to have more data from the further angles , over sampled these data to make the distribution look like the image below.

First image is the data set with only steering angle distribution with center images only.
![alt text][image3]

Second image describes the data that was augmented with left , right and over sampling. Also, the data with steering angles of 0 was under sampled.
![alt text][image4]






After the collection process, I had 29786 number of data points. I then preprocessed this data by by cropping the image.
![alt text][image1]

Cropped image
![alt text][image2]


I finally randomly shuffled the data set and put Y% of the data into a validation set.


Adam optimizer was used with gradient clipping enabled (to keep the gradients between -1 and 1, regularizer).

### 4. Result

Link below shows the result of this trained model.

[![Self Driving Car](https://img.youtube.com/vi/GLSbGck3k3Q/0.jpg)](https://youtu.be/GLSbGck3k3Q "Self Driving Car")
