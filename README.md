# CNN-HowManyFingers
Count how many fingers are being held up, via a convolutional neural network (CNN) implemented in Keras + TensorFlow + OpenCV.

Key Requirements: Python 3+, Keras 2+, TensorFlow 1+, OpenCV 2+

## Contents
* **application.py** - application to used to collect data and predict number of fingers on the fly
* **trainModel.ipynb** - jupyter notebook used for training model
* **images.tgz** - compressed tarball of complete training and validation data
* **imgs** - directory of example images used in this README
* [**model_6cat.h5**](https://drive.google.com/file/d/0B5sZ8q5iqYbtZjRRRW1SUVl2SlU/view?usp=sharing) - pretrained model

The pretrained model used by the application is can be downloaded from google drive, linked above.

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
