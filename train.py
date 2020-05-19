# your implementation goes here
import random
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, ZeroPadding2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt



IMG_HEIGHT=256
IMG_WIDTH = 256
IMG_CHANNELS = 3

seed = 42
random.seed = seed
np.random.seed = seed

parser = argparse.ArgumentParser(description='Load training data')
parser.add_argument('--in_train', type=str, help='path to training image')
parser.add_argument('--in_gt', type=str, help='path to ground truth image')
parser.add_argument('--out_crop', type=str, help='Output dir for sliced 256x256 images')
parser.add_argument('--out_train', type=str, help='Output dir for final training images')
parser.add_argument('--out_test', type=str, help='Output dir for final testing images')

args = vars(parser.parse_args())


train_image = args['indir_train']
gt_image = args['indir_gt']

img_before = cv2.imread(train_image)
gt = cv2.imread(gt_image)

print("Shape of input image",img_before.shape)
print("Shape of ground truth",gt.shape)

# crop image into smaller size
rows=0
cols=0
k=0
while (rows+256 < img_before.shape[0]):
    cols=0
    while (cols+256 < img_before.shape[1]):
        img_crop = img_before[rows:rows+256, cols:cols+256, :]
        cv2.imwrite(args['outdir_crop']+"/crop%s.jpg" %k, img_crop)
        gt_crop = gt[rows:rows+256, cols:cols+256]
        cv2.imwrite(args['outdir_crop']+"/gt_crop%s.jpg" %k, gt_crop)
        cols+=256
        k+=1
    img_crop = img_before[rows:rows+256,cols:img_before.shape[1],:]
    img_crop = cv2.resize(img_crop, (256,256))
    cv2.imwrite(args['outdir_crop']+"/crop%s.jpg"%k, img_crop)
    gt_crop = gt[rows:rows+256,cols:img_before.shape[1]]
    gt_crop = cv2.resize(gt_crop, (256,256))
    cv2.imwrite(args['outdir_crop']+"/gt_crop%s.jpg"%k, gt_crop)
    k+=1
    rows+=256

if (rows+256>img_before.shape[0]):
    cols=0
    while (cols+256 < img_before.shape[1]):
        img_crop = img_before[rows:img_before.shape[0], cols:cols+256, :]
        img_crop = cv2.resize(img_crop, (256,256))
        print(k)
        cv2.imwrite(args['outdir_crop']+"/crop%s.jpg" %k, img_crop)
        gt_crop = gt[rows:img_before.shape[0], cols:cols+256]
        gt_crop = cv2.resize(gt_crop, (256,256))
        cv2.imwrite(args['outdir_crop']+"/gt_crop%s.jpg" %k, gt_crop)
        cols+=256
        k+=1
    img_crop = img_before[rows:img_before.shape[0],cols:img_before.shape[1],:]
    img_crop = cv2.resize(img_crop, (256,256))
    cv2.imwrite(args['outdir_crop']+"/crop%s.jpg"%k, img_crop)
    gt_crop = gt[rows:img_before.shape[0],cols:img_before.shape[1]]
    gt_crop = cv2.resize(gt_crop, (256,256))
    cv2.imwrite(args['outdir_crop']+"/gt_crop%s.jpg"%k, gt_crop)

X = np.zeros((k, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y = np.zeros((k, IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)

i = 0
j=0
while (i<k):
    img = cv2.imread(args['outdir_crop']+"/crop%s.jpg"%i)
    gt_img = cv2.imread(args['outdir_crop']+"/gt_crop%s.jpg"%i)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    gt_img = gt_img/255
    if(np.mean(gt_img)<1):
        X[j] = img
        Y[j] = gt_img
        j+=1
    i+=1
X_t = np.delete(X ,np.s_[j:k],0)
Y_t = np.delete(Y, np.s_[j:k],0)

X_train, X_test, y_train, y_test = train_test_split(X_t, Y_t, random_state=42, test_size=0.1)

n_train = X_train.shape[0]
n_test = X_test.shape[0]

i = 0
while (i<n_train):
    cv2.imwrite(args['outdir_train']+"/train%s.jpg"%i, X_train[i])
    i+=1

i=0
while (i<n_test):
    cv2.imwrite(args['outdir_test']+"/test%s.jpg"%i, X_test[i])
    i+=1

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

#p1 = ZeroPadding2D((1,1)) (s)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',padding = 'same') (s)

#p2 = ZeroPadding2D((1,1)) (c1)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal',padding = 'same') (c1)

m1 = MaxPooling2D((2, 2)) (c2)

#p3 = ZeroPadding2D((1,1)) (m1)
c3 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal',padding = 'same') (m1)

u1 = UpSampling2D(size=(2, 2), interpolation='nearest') (c3)

#p4 = ZeroPadding2D((2,2)) (u1)
c4 = Conv2D(2, (5, 5), activation='relu', kernel_initializer='he_normal',padding = 'same') (u1)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c4)

model = Model(inputs=[inputs], outputs=[outputs])
adam  = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer='adam', loss='binary_crossentropy')
print(model.summary())

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-fcnn.h5', monitor='loss', mode='auto', verbose=1, save_best_only=True)
results = model.fit(X_train, y_train, validation_split=0.1, batch_size=8, epochs=50,
                    callbacks=[earlystopper, checkpointer])

# summarize history for loss
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
