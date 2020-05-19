# your implementation goes here
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import label

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, ZeroPadding2D, UpSampling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers


import cv2
import argparse


class automaticmaplabelling():
    def __init__(self,modelPath,full_chq,imagePath,width,height,channels):
        print (modelPath)
        print(imagePath)
        print(width)
        print(height)
        print(channels)
        self.modelPath=modelPath
        self.full_chq=full_chq
        self.imagePath=imagePath
        self.IMG_WIDTH=width
        self.IMG_HEIGHT=height
        self.IMG_CHANNELS=channels
        self.model = self.FCNN()

    def FCNN(self):
        # Load FCNN model
        model = load_model(self.modelPath)
        model.summary()
        return model

    def prediction(self):
        img=cv2.imread(self.imagePath)
        #img=np.expand_dims(img,axis=-1)
        x_test = np.zeros((1, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        img=resize(img,(self.IMG_HEIGHT,self.IMG_WIDTH),mode='constant',preserve_range=True)
        x_test[0]=img
        preds_test= self.model.predict(x_test, verbose=1)

        preds_test = (preds_test > 0.5).astype(np.uint8)
        mask=preds_test[0]
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j] == 1:
                    mask[i][j] = 255
                else:
                    mask[i][j] = 0
                    x_test[0][i][j][:] = 0
        merged_image = cv2.merge((mask,mask,mask))
        return x_test[0],mask

def main():
    parser = argparse.ArgumentParser(description='Load test image')
    parser.add_argument('--in', type=str, help='path to test image')
    parser.add_argument('--out', type=str, help='Output dir for predicted images')
    parser.add_argument('--model_path', type=str, help='path to model')
    args = vars(parser.parse_args())
    #print(args['model_path'])
    test_image_name = args['indir_test']
    automaticmaplabellingobj= automaticmaplabelling(args['model_path'],True,test_image_name,256,256,3)
    test_img,mask = automaticmaplabellingobj.prediction()
    cv2.imwrite(args['outdir_pred']+"/mask.jpg",mask)
    cv2.imwrite(args['outdir_pred']+"/overlap.jpg", test_img)

if __name__ == "__main__":
    main()
