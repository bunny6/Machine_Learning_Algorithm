#imported tensorflow
#ImageDataGenerator is a tool that will applt transformation.
import tensorflow as tf                                 
from keras.preprocessing.image import ImageDataGenerator 

#checking the version of tensorflow
tf.__version__              

#Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,       #rescale is used for scaling.
                                   shear_range = 0.2,      #shear_range specifies the angle of the slant in degrees.
                                   zoom_range = 0.2,       #to zoom in and zoom out.
                                   horizontal_flip = True) #A horizontal filp will be done
training_set = train_datagen.flow_from_directory('dataset/training_set/',  #here we specify the path
                                                 target_size = (64, 64),   #the size of the image we will feed to the Neural network.
                                                 batch_size = 32,          #here we specigy the size of the batch.
                                                 class_mode = 'binary')    #here we specify the class_mode, binary or category.


#Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


#Initialising the CNN
cnn = tf.keras.models.Sequential()               

#creating the convolutional layer.
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3])) 

#Pooling layer.
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) #creating the pooling layer.

#Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))  
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Flattening
cnn.add(tf.keras.layers.Flatten())   #flattening means converting rows into sinle column. The required input format to the ann.

#Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))   #initialising the ann.

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  #creating the output layer. Unit 1 because we have one output neuron.

#Training the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])  #compiling the cnn. Optimizer is stcoustic gradient descent method.

#Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)  

#Making a single prediction
import numpy as np                  
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))   #loading the image by using load_img function.
test_image = image.img_to_array(test_image)   #converting the image to numpy array.
test_image = np.expand_dims(test_image, axis = 0)   #as we trained the cnn with batch=32. Here we have to create a fake dimention and axis=0 means where we want too add the dimensionality.
result = cnn.predict(test_image)  #predicting
training_set.class_indices  #to get the indices we use class_indices. It will return what correspondence to which class.
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'

#printing the result
print(prediction)   

