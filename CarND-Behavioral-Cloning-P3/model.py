import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.optimizers import Adam

from keras.layers.core import Dense, Activation, Flatten,Dropout,Lambda
from keras.layers.convolutional import Convolution2D,Cropping2D
from keras.layers.pooling import MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential


"""
Data Augentation 
"""
def data_augmentation():
    augmented_data = []
    augmented_steer = []
    perturb = 0.0
    for d,s in zip(logs.center,logs.steering):
        # if not zero flip
        im = Image.open("./data/"+d)
        if(s != 0.0):
            #im.thumbnail((160,320), Image.ANTIALIAS)
            augmented_data.append(np.array(im.transpose(Image.FLIP_LEFT_RIGHT)))
            augmented_steer.append(s*-1.0)
        if(s == 0.0):
            import random
            if(random.random() < 0.2):
                augmented_data.append(np.array(im))
                augmented_steer.append(s)
        else:
            augmented_data.append(np.array(im))
            augmented_steer.append(s)

    # left steering angles
    correction_factor = 0.2
    for d,s in zip(logs.left,logs.steering):
        old_s =s 
        s = s +correction_factor

        im = Image.open("./data/"+d.strip())
        if(old_s == 0.0):
            import random
            if(random.random() < 0.2):
                augmented_data.append(np.array(im.transpose(Image.FLIP_LEFT_RIGHT)))
                augmented_steer.append((s+perturb)*-1.0)
            if(random.random() < 0.2):
                augmented_data.append(np.array(im))
                augmented_steer.append(s+perturb)
        else:
            augmented_data.append(np.array(im.transpose(Image.FLIP_LEFT_RIGHT)))
            augmented_steer.append((s+perturb)*-1.0)
            augmented_data.append(np.array(im))
            augmented_steer.append(s+perturb)
            import random
            if(random.random() < np.abs(s)):
                augmented_data.append(np.array(im.transpose(Image.FLIP_LEFT_RIGHT)))
                augmented_steer.append((s+perturb)*-1.0)
            if(random.random() < np.abs(s)):
                augmented_data.append(np.array(im))
                augmented_steer.append(s+perturb)

    #right steering angle
    for d,s in zip(logs.right,logs.steering):
        old_s =s 
        s = s -correction_factor

        im = Image.open("./data/"+d.strip())
        if(old_s == 0.0):
            import random
            if(random.random() < 0.2):
                augmented_data.append(np.array(im.transpose(Image.FLIP_LEFT_RIGHT)))
                augmented_steer.append((s+perturb)*-1.0)
            if(random.random() < 0.2):
                augmented_data.append(np.array(im))
                augmented_steer.append(s+perturb)
        else:
            imf,sf = flip_image(im,s)
            augmented_data.append(np.array(im.transpose(Image.FLIP_LEFT_RIGHT)))
            augmented_steer.append((s+perturb)*-1.0)
            augmented_data.append(np.array(im))
            augmented_steer.append(s+perturb)
            import random
            if(random.random() < np.abs(s)):
                augmented_data.append(np.array(im.transpose(Image.FLIP_LEFT_RIGHT)))
                augmented_steer.append((s+perturb)*-1.0)
            if(random.random() < np.abs(s)):
                augmented_data.append(np.array(im))
                augmented_steer.append(s+perturb)

    return augmented_data,augmented_steer

import sklearn
from PIL import Image

"""
Contrast Increase using HSV values.
"""
def process_data(image):
    im2 = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV)
#     im2 = cv2.cvtColor(im2,cv2.COLOR_HSV2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     im = clahe.apply(im2)
    #cropping the image
    im = im2
    im = np.array(im)[50:120]
    return im
    
    
"""
Flipping image and steering
"""
def flip_image(image,steer,perturb = False):
    # Adding a small perturb to the steer ( noise )
    if(perturb):
        steer = steer +np.random.normal(0.001,0.001)
    return image.transpose(Image.FLIP_LEFT_RIGHT),steer*-1.0
"""
Generator to generate batches of training data
"""
def generator(X_train,y_train,batch_size=128):
    augmented_data = X_train 
    augmented_steer = y_train 
    X_train = np.array(augmented_data)
    y_train = np.array(augmented_steer)
    total = len(X_train)
    n_iter = int(total/batch_size)
    X_train,y_train = sklearn.utils.shuffle(X_train,y_train)
    while 1:
        for i in range(n_iter): 
            #print(i)
            yield X_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
        

def main():
    augmented_data,augmented_steer = data_augmentation()
    X_train = np.array(augmented_data)
    y_train = np.array(augmented_steer)




    model = Sequential()
    model.add(Lambda(lambda x : x/255.0-0.5 ,input_shape=(160, 320,3)))
    model.add(Cropping2D(cropping=((70,20),(0,0))))
    model.add(Convolution2D(32,3,3))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64,3,3))
    #model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128,3,3))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.8))
    model.add(Activation('relu'))

    model.add(Dense(84))
    model.add(Dropout(0.8))
    model.add(Activation('relu'))

    model.add(Dense(1))
    #gradient clipping
    opt = Adam(clipvalue=1.0)
    model.compile(optimizer=opt,loss='mse')

    # with generator
    #train_generator = generator(X_train,y_train,batch_size=128)
    #model.fit_generator(train_generator,samples_per_epoch=20000 , nb_epoch=5)

    #without generator
    model.fit(X_train,y_train,batch_size=128,validation_split=0.2,nb_epoch=5,shuffle=True)

    model.save("model.h5")

if __name__ == "__main__":
    main()