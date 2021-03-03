import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator #image generator label data based on the dir the image in contained in
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
import os
import zipfile

train_normal_dir = os.path.join('C:\\Users\\NAME\\FOLDER\\data\\COVID\\chest_xray\\train\\NORMAL')
train_covid_dir = os.path.join('C:\\Users\\NAME\\FOLDER\\data\\COVID\\chest_xray\\train\covid')
test_normal_dir = os.path.join('C:\\Users\\NAME\\FOLDER\\data\\COVID\\chest_xray\\val\\NORMAL')
test_covid_dir = os.path.join('C:\\Users\\NAME\\FOLDER\\data\\COVID\\chest_xray\\val\covid')

train_normal_names = os.listdir(train_normal_dir)
train_covid_names = os.listdir(train_covid_dir)
test_normal_names = os.listdir(test_normal_dir)
test_covid_names = os.listdir(test_covid_dir) 

print('total train normal chest xray: ', len(os.listdir(train_normal_dir)))
print('total train COVID chest xray:', len(os.listdir(train_covid_dir)))
print('total test normal chest xray: ', len(os.listdir(test_normal_dir)))
print('total test COVID chest xray: ', len(os.listdir(test_covid_dir)))

# Building the model and making it more dense.
model = tf.keras.models.Sequential([
  
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
  
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
  
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
  
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
  
    # The fifth convolution
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

  
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'), # 512 neuron hidden layer
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for ('normal') clas and 1 for ('covid') class
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])

# Before fitting the model for training, creating data generators to read images instead of reading each one by one,
# making the process faster.
train_datagen = ImageDataGenerator(rescale = 1/255)
test_datagen = ImageDataGenerator(rescale = 1/255)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\NAME\\FOLDER\\data\\COVID\\chest_xray\\train',
    target_size = (300,300),
    batch_size = 128,
    class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    'C:\\Users\\NAME\\FOLDER\\data\\COVID\\chest_xray\\val',
    target_size = (300, 300),
    batch_size = 128,
    class_mode = 'binary'
)

# The learning part of the program. The computer is given a training set of about 5,400 images for it to learn from.
history = model.fit(
    train_generator,
    steps_per_epoch = 10,
    epochs = 10,
    validation_data = validation_generator
)

# Test N is for people who are normal, while Test C is for people who have the novel coronavirus.
testN = 'C:\\Users\\NAME\\FOLDER\\data\\COVID\\chest_xray\\test\\NORMAL\\IM-0001-0001.jpeg'
testC = 'C:\\Users\\NAME\\FOLDER\\data\\COVID\\chest_xray\\test\\COVID\\person1_virus_6.jpeg'


img = image.load_img(testC, target_size=(300,300))
plt.imshow(img, cmap = 'gray')

x = image.img_to_array(img)
x = np.expand_dims(x, axis =0)

images = np.vstack([x])
classes = model.predict(images, batch_size = 10)
print(classes[0])
if classes[0]> 0.5:
    print('Patient has COVID-19')
    plt.imshow(img)
else:
    print('Patient is normal')
    plt.imshow(img)

# Graphing the accuracy of the AI model
fig, ax = plt.subplots(1,2)
fig.set_size_inches(10,5)

metric = ['accuracy', 'loss']
for i, m in enumerate(metric):
  ax[i].plot(history.history[m])
  ax[i].plot(history.history['val_'+ m])
  ax[i].set_title('Model {}'.format(m))
  ax[i].set_xlabel('epochs')
  ax[i].set_ylabel('m')
  ax[i].legend(['train', 'validation'])

# Loading a dataset that has not been seen by the computer before
test_datagen = ImageDataGenerator(rescale = 1/255)

test_generator = test_datagen.flow_from_directory(
    'C:\\Users\\NAME\\FOLDER\\data\\COVID\\chest_xray\\val',
    target_size = (300, 300),
    batch_size = 128,
    class_mode = 'binary'
)

eval_result = model.evaluate_generator(test_generator, 16)
print('loss rate at evaluation data :', eval_result[0])
print('accuracy rate at evaluation data :', eval_result[1])

# At this point, the model is now ready to make a prediction by itself. It has finished training and will now make a prediction.
# The user can input an image and the computer will detect whether the x-ray is normal or if the patient has COVID-19.

from google.colab import files
uploaded = files.upload()

for fn in uploaded.keys():
  # predict images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(300,300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis =0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size = 10)
  print(classes[0])
  if classes[0]> 0.5:
    print(fn + ' is COVID')
    plt.imshow(img)
  else:
    print(fn + 'is normal')
    plt.imshow(img)

import pandas as pd
from xlrd import open_workbook
import numpy as np
from sklearn.linear_model import LinearRegression

file_name =  'C:\\Users\\NAME\\DEMOGRAPHICDATASETFOLDER\\COVID.xlsx'
sheet =  'Line-list'

df = pd.read_excel(io=file_name, sheet_name=sheet)

x = df[["Covid_Positive", "age"]].to_numpy()
y = df[["gender"]].to_numpy()

model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)

print('confidence of model:', r_sq)





