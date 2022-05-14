#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd # read csv
import matplotlib.pyplot as plt #creates figure and work like matlab
import seaborn as sns
import os #interact with os functions
for dirname, _, filenames in os.walk('C:\\Users\\higar\\fruits-360_dataset\\fruits-360'): #directory tree
    for filename in filenames: #each filename
        print(os.path.join(dirname, filename)) #print dirname and filename together


# In[11]:


from keras.models import Sequential #creating deep models
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


# In[12]:


train_path = "C:\\Users\\higar\\fruits-360_dataset\\fruits-360\\Training\\"
test_path = "C:\\Users\\higar\\fruits-360_dataset\\fruits-360\\Test\\"


# In[14]:


img = load_img(train_path + "Quince\\r_305_100.jpg") #keras
plt.imshow(img)
plt.axis("on")
plt.show()


# In[15]:


img = img_to_array(img) #numpy array of pixel data
img.shape #rows,column,channel, gray null


# In[16]:


model = Sequential() 
model.add(Conv2D(128, 3, activation="relu", input_shape=(100,100,3))) #2D convolution layer: tensor of outputs
model.add(MaxPooling2D()) #reduce spatial dimension output volume
model.add(Conv2D(64, 3, activation="relu")) #kernel dimension
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.50)) #sets input units 0. Frequency rate each step during training time
model.add(Flatten()) #multidimensional input tensor to single tensor
model.add(Dense(5000, activation = "relu")) #inputs preveous neurons, input data parameter
model.add(Dense(1000, activation = "relu")) #activation argument or layer
model.add(Dense(131, activation = "softmax")) #softmax: last dimension input
model.summary()


# In[17]:



model.compile(loss="categorical_crossentropy", optimizer = "SGD", metrics = ["accuracy"])


# In[18]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                  shear_range = 0.3,
                  horizontal_flip=True,
                  vertical_flip=False,
                  zoom_range = 0.3
                  )
test_datagen  = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(100,100),
                                                    batch_size = 32,
                                                    color_mode= "rgb",
                                                    class_mode = "categorical")
test_generator = test_datagen.flow_from_directory(test_path,
                                                    target_size=(100,100),
                                                    batch_size = 32,
                                                    color_mode= "rgb",
                                                    class_mode = "categorical")


# In[19]:


hist = model.fit_generator(generator = train_generator, #train model
                   steps_per_epoch = 50,
                   epochs = 50,
                   validation_data = test_generator,
                   validation_steps = 50)


# In[20]:


from keras.models import load_model

model.save("Fruitmodel.h5")
import tensorflow as tf 
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
test_generator.classes[22687] #labels


# In[24]:


predictions = probability_model.predict(test_generator) #model created, fitted with trained data and now prediction


# In[28]:


b=predictions[8]
print(b)


# In[26]:


np.argmax(b)


# In[30]:


folder = 'C:\\Downloads\\cherry_test.jpg'


# In[37]:


import os 
test_generator.classes[222]


# In[23]:


import os

folder = 'C:\\Users\\higar\\fruits-360_dataset\\fruits-360\\Test'

sub_folders = [name for name in os.listdir(folder)] #names of fruits

print(sub_folders[33])


# In[1]:


import os

folder = 'C:\\Users\\higar\\fruits-360_dataset\\fruits-360\\Test'

sub_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]

print(sub_folders[2])


# In[ ]:





# In[ ]:




