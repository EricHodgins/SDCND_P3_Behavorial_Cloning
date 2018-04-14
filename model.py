import numpy as np
import csv
import cv2

def open_files_return_samples(folder_path):
    samples = []
    with open(folder_path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            sample = []
            center_path = folder_path + '/IMG/' + line[0].split('/')[-1]
            left_path = folder_path + '/IMG/' + line[1].split('/')[-1]
            right_path = folder_path + '/IMG/' + line[2].split('/')[-1]
            angle = line[3]
            sample.append(center_path)
            sample.append(left_path)
            sample.append(right_path)
            sample.append(angle)
            samples.append(sample)
            
    return samples

samples = []
s0 = open_files_return_samples('./data')
s0.pop(0) # Remove Header line from Udacity's data

samples = s0

# Remove 0 valued steering angles
print(len(samples))
filtered = []
zero_count = 0
for index, s in enumerate(samples):
    if float(s[3]) == 0.0 and zero_count < 2000:
        zero_count += 1
        continue
    filtered.append(s)
print(len(filtered))
samples = filtered


# Split the data into Training and Validation Sets
from sklearn.model_selection import train_test_split
import sklearn

train_samples, validation_samples = train_test_split(samples, test_size=0.2)



# Creat Keras Generator functions for training and validation 
from random import randint

BATCH_SIZE = 32
AUGMENTED_PICS_ADDED_TO_SAMPLE = 2
STEERING_CORRECTION = 0.18

def generator(samples, batch_size=32):
    num_samples = len(samples) // batch_size
    while True:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_path = batch_sample[0]
                left_path = batch_sample[1]
                right_path = batch_sample[2]
                
                # Center Image and Flip
                center_image = cv2.imread(center_path)
                center_flipped_image = cv2.flip(center_image, 1)
                center_angle = float(batch_sample[3])
                center_angle_inverse = -center_angle
                
                # Left Image and Flip
                left_image = cv2.imread(left_path)
                left_flipped_image = cv2.flip(left_image, 1)
                left_angle = center_angle + STEERING_CORRECTION
                left_angle_inversed = -center_angle - STEERING_CORRECTION
                
                # Right Image and Flip
                right_image = cv2.imread(right_path)
                right_flipped_image = cv2.flip(right_image, 1)
                right_angle = center_angle - STEERING_CORRECTION
                right_angle_inversed = -center_angle + STEERING_CORRECTION
                
                pick = randint(0, 2)
                
                if pick == 0:
                    images.append(center_image)
                    images.append(center_flipped_image)
                    angles.append(center_angle)
                    angles.append(center_angle_inverse)
                elif pick == 1:
                    images.append(left_image)
                    images.append(left_flipped_image)
                    angles.append(left_angle)
                    angles.append(left_angle_inversed)
                else:
                    images.append(right_image)
                    images.append(right_flipped_image)
                    angles.append(right_angle)
                    angles.append(right_angle_inversed)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

## MODEL ARCHITECTURE


from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Lambda, Cropping2D, Dropout, MaxPooling2D
from keras.layers import Conv2D
from keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

model.add(Cropping2D(cropping=((70,25), (0, 0))))

model.add(Conv2D(24, (5, 5), strides=(2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(36, (5, 5), strides=(2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(48, (5, 5), strides=(2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(64, (3, 3), strides=(2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(64, (3, 3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

checkpoint = ModelCheckpoint('model-mid.h5', monitor='val_loss', verbose=0, save_best_only=False, 
                              save_weights_only=False, mode='auto', period=1)

model.fit_generator(train_generator, 
                    steps_per_epoch=len(train_samples)*AUGMENTED_PICS_ADDED_TO_SAMPLE//BATCH_SIZE, 
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)*AUGMENTED_PICS_ADDED_TO_SAMPLE//BATCH_SIZE, 
                    epochs=5, verbose=1, shuffle=True, callbacks=[checkpoint])

model.save('model.h5')



























