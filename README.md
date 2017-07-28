# CNN-HowManyFingers
Count how many fingers are being held up, via a convolutional neural network (CNN) implemented in Keras + TensorFlow + OpenCV.

The model is able to surpass human perform in the case that the human is severely concussed or inebriated.

Key Requirements: Python 3+, Keras 2+, TensorFlow 1+, OpenCV 2+

## Contents
* **application.py** - application to used to collect data and predict number of fingers on the fly
* **trainModel.ipynb** - jupyter notebook used for training model
* **images.tgz** - compressed tarball of complete training and validation data
* [**model_6cat.h5**](https://drive.google.com/file/d/0B5sZ8q5iqYbtZjRRRW1SUVl2SlU/view?usp=sharing) - pretrained model

The pretrained model used by the application is can be downloaded from google drive, linked above.

## Demo
Demos with and without binary mask visible. Noise in images is coming from camera.
![](https://i.imgur.com/nbpeRgg.gif)
![](https://i.imgur.com/Ref2OVT.gif)


## Usage
Simply run `application.py` and hold your fingers up in the highlighted region of interest (ROI).
The model performs best when provided a plain background without many features.

### Hotkeys
* The ROI on the screen can be moved using the `i`, `j`, `k`, and `l` keys.
* Show the binary mask being applied to the ROI by toggling the `b` key.
* Quit the application by pressing the `q` key.

### Taking Data
To collect data, you must first select the target class 0 through 5 by hitting the corresponding key on your keyboard.
Once you have selected your target class, you can start/stop collecting data by toggling the `s` key. When collecting 
data the outline of the ROI will turn green and turn red again when collection is stopped.

# Feature Input

Images are captured within a ROI from the webcam using OpenCV. To help simplify the inputs that are analyzed by the CNN, a binary mask is applied to highlight the hands edges. The binary mask is defined by grayscaling and blurring the image and then applying thresholding as shown below:

```
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7,7), 3)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

A dataset is collected within the application by holding up 0 to 5 digits at different positions and orientations within the application's ROI. A training set of ~1500 images and validation set of ~600 images for each case is used for training the CNN. 

![](https://i.imgur.com/ylMUgE5.png)

# Convolution Neural Net

The CNN used for this projects consists of 4 convolutional layers with 3x3 kernels, RELU activations and integer multiples of 32 filters in each layer. Between each convolutional layer MaxPooling is  appled to reduce the models dimensionality. The feature maps produced by the convolutions are passed to a dense layer with 512 nodes and RELU activation before being fed to a softmax output layer with 6 nodes, defining each class.

```
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(300,300,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='sigmoid'))
```

# Training and Performance

The model is trained with TensorFlow backend using a NVIDIA GeForce Titan X Pascal for 40 epochs using batches of 128 images each. The training inputs are augmented with small randomized zooms, rotations and translations for each instance in training. Mirroring is also applied to the input images to not bias the results to any particular handedness. A testing performance of greater than 99% accuracy is achieved, surpassing human perform in the case that the human is severely concussed or inebriated.

![](https://i.imgur.com/tWWoyUO.png)

# Generalizability
Interestingly, the model seems to generalize quite well to new combinations of digits that were never before seen in the training datasets.

![](https://i.imgur.com/jTP4I5C.gif)

# Error analysis 
It occasionally happens that during the (random) image augmentation that is applied by our generator, one of the hands fingers is obscured from the image. These cases typically manifest as misclassification "error" in our accuracy but should not be a concern as the prediction made by the network is typically the same conclusion that would have been made by a human with the partial information presented them.

<img src="https://i.imgur.com/s0exjr8.png" width="500">

The confusion matrix shows very strong performance generally, however the model shows some difficulty in correctly counting four fingers. This misclassification seems most prevalent when the hand is angled and could likely be improved with additional data, augmentation and perhaps additional model complexity. For the purposes of this prototype, this caveat is deemed acceptable but could be improved in the future.

![](https://i.imgur.com/gJczLM5.png)

Lastly, the model's predictions are observed to be sensitive to shadows and lighting in the region of interest. 
The model is also sensitive to features in the background. Refined image processing could be more robust to the lighting
and able to isolate the hands features better. Alternatively, additional training data or augmentation with 
busy backgrounds and various lighting might make the model more robust.
