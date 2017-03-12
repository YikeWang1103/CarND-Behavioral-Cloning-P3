import csv
import cv2
import numpy as np
import sklearn
import random

# load data
lines=[]
with open('/home/andy/Udacity_Project/CarND-Behavioral-Cloning-P3/Drive_Data/NormalDrive/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# split the data for training and validation        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines,test_size=0.2)

# define generator to save GPU memories
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images=[]
            angles=[]
            for batch_sample in batch_samples:
                
                centerImageName = batch_sample[0]
                centerImage = cv2.imread(centerImageName)
                centerAngle = float(batch_sample[3])
                images.append(centerImage)
                angles.append(centerAngle)
                
                leftImageName = batch_sample[1]
                leftImage = cv2.imread(leftImageName)
                leftAngle = float(batch_sample[3])+0.5
                images.append(leftImage)
                angles.append(leftAngle)  
                
                rightImageName = batch_sample[2]
                rightImage = cv2.imread(rightImageName)
                rightAngle = float(batch_sample[3])-0.5
                images.append(rightImage)
                angles.append(rightAngle)
                
                augmented_images, augmented_measurements = [],[]
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train,y_train)
            
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print('Modules loaded.')

# define model architecture
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: x/255.-1.,
                 input_shape=(160,320,3),
                 output_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(12, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 3, 3, activation="relu"))
model.add(Convolution2D(60, 3, 3, activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

# model training
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples), 
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples),
                                     nb_epoch=7)

#save the model
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()