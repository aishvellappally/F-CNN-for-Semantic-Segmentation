# Image Segmentation

Image segmentation is done using a Fully Connected Neural Network as per given requirements.

## Packages

Required packages are in requirement.txt

## Usage
The folder contains trained model in "model-fcnn.h5". If required to train again, give inputs as per following example for train.py :
```
(new_env) pc-136-30:ML_exercise aiswarya$ python train.py
--in_train "images/rgb.png" --in_gt "images/gt.png"
--out_crop "images/crop" --out_train "images/train"
--out_test "images/test"
```
in_train - relative path to input image \
in_gt - relative path to ground truth images \
out_crop - relative path to folder where sliced input and ground truth  images of size 256x256 will be saved \
out_train - relative path to folder to save final training images \
out_test - relative path to folder to save final test images

The generated 256x256 images, final training and test images are saved in folders crop, train and test respectively.

For prediction, specify the relative path to test image, relative path to saved model and path to prediction folder as per following example:

```
(new_env) pc-136-30:ML_exercise aiswarya$ python predict.py --in "images/test/test1.jpg" --out "images/pred" --model_path "model-fcnn.h5"
```
The prediction of given test image will be saved in folder named "pred". One example has already been tested and saved in the pred folder.

Train_loss.png has the training and validation curve plot.

## Note

1. Training must be done on RGB image. It can be of any size. The program takes care of GPU size restriction.
2.Test image must also be RGB, and can be any size.
3. All paths must be relative.
