from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import Conv2D,Activation, MaxPooling2D,Dense,Flatten, Dropout, BatchNormalization
from keras.optimizers import Adamax, Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import regularizers
from keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import h5py

import os
import numpy as np
from PIL import Image
import random

from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from resnet50 import MyResNet50


#Simple CNN model (does not converge)
def Create_Model():
    model = Sequential()

    #First Block
    model.add(Conv2D(32, (3, 3), 
                     padding='same',
                     input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))   
    model.add(Dropout(0.3))
    
    #Second Block
    model.add(Conv2D(64, (3, 3),  padding='same'))  
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    
    #Third Block
    model.add(Conv2D(128, (3, 3), padding='same'))  
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))   
    model.add(Dropout(0.3))

    #Fourth Block
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same')) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))  
    model.add(Dropout(0.3))
    
    #Fifth Block
    model.add(Conv2D(512, (3, 3), padding='same')) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))  
    model.add(Dropout(0.3))
    
    #  Flatten and Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))   
    model.add(Dropout(0.15))  
    
    # Sigmoid Classifier (can swap for softmax for multiple classes)
    model.add(Dense(1))
    model.add(Activation('sigmoid'))        
	
    # compile model
    opti = Adam(learning_rate=0.001)
    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy'])

    #  Display model
    model.summary()

    return model 

#Pretrained EfficientNetB3 model (78% accuracy)
def Create_EfficientNetB3_Model():
   
   base = tf.keras.applications.efficientnet.EfficientNetB3(include_top = False, weights='imagenet', input_shape = (224, 224, 3), pooling='max')

   model = Sequential([
      base,
      BatchNormalization(),
      Dense(512),
      Activation('relu'),
      Dropout(rate=0.45),#0.5 last test
      Dense(1),
      Activation('sigmoid')
   ])

   opti = Adamax(learning_rate=0.0001)
   model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy'])

   model.summary()

   return model

#Custom ResNet50 model (75% accuracy)
def Create_My_ResNet50_Model():
    
    model = MyResNet50(input_shape=(224, 224, 3), drop_rate=0.5, l2_r=0.007)
    
    opti = Adam()
    
    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    return model

#decreases learning rate after certain epochs
def learnR_schedule(epoch):
    lr = 0.0005
    if epoch >= 20:
        lr *= 0.001
    if epoch >= 15:
        lr *= 0.01
    elif epoch >= 10:
        lr *= 0.01
    elif epoch >= 5:
        lr *= 0.1
    return lr

def learnR_schedule_ResNet50(epoch):
    lr = 0.0001
    if epoch >= 20:
        lr *= 0.001
    if epoch >= 15:
        lr *= 0.05
    elif epoch >= 10:
        lr *= 0.05
    elif epoch >= 5:
        lr *= 0.01
    return lr

#decreases learning rate after certain epochs
#def learnR_schedule(epoch):
#    lr = 0.0001*(0.85**epoch)
#    
#    return lr
    
def load_training_data():
    #load training data
    train_dir = 'data2/train/'
    benign_dir = os.path.join(train_dir, 'benign')
    malignant_dir = os.path.join(train_dir, 'malignant')
    benign_files = os.listdir(benign_dir)
    malignant_files = os.listdir(malignant_dir)

    combined_data = []
    labels = []
    


    #loop through benign directory
    for i, file in enumerate(benign_files):
        if i >= 1197:
            break
        image = Image.open(os.path.join(benign_dir, file))
        image = image.resize((224, 224))
        image = np.array(image)/255
        combined_data.append(image)
        labels.append(0)
        if i <= 4:
            plt.subplot(2, 5, i+1)
            plt.imshow(image)
            plt.axis('off')
            plt.title('Benign')

    #loop through malignant directory
    for i, file in enumerate(malignant_files):
        image = Image.open(os.path.join(malignant_dir, file))
        image = image.resize((224, 224))
        image = np.array(image)/255
        combined_data.append(image)
        labels.append(1)
        if i <= 4:
            plt.subplot(2, 5, i+6)
            plt.imshow(image)
            plt.axis('off')
            plt.title('Malignant')
    
    #save file with sample images
    plt.savefig('training_images.png')
        
    print('Images total: ', len(combined_data))

    return combined_data, labels

def load_test_data():
    #load test data
    test_dir = 'data2/test/'
    benign_dir = os.path.join(test_dir, 'benign')
    malignant_dir = os.path.join(test_dir, 'malignant')
    benign_files = os.listdir(benign_dir)
    malignant_files = os.listdir(malignant_dir)

    combined_data = []
    labels = []

    #loop through benign directory
    for i, file in enumerate(benign_files):
        if i >= 300:
            break
        image = Image.open(os.path.join(benign_dir, file))
        image = np.array(image)/255
        combined_data.append(image)
        labels.append(0)

    #loop through malignant directory
    for file in malignant_files:
        image = Image.open(os.path.join(malignant_dir, file))
        image = np.array(image)/255
        combined_data.append(image)
        labels.append(1)
    
    print('Images total: ', len(combined_data))

    return combined_data, labels



def Train_Model(model, name):
    
    #load data for training
    X, y = load_training_data()

    #data splitted, 20% allocated to validation as recommended
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    
    #data augmentation technique to artificially expand data set
    t_datagen = ImageDataGenerator(
        featurewise_center= True,
        featurewise_std_normalization = True,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        height_shift_range=0.2,
        width_shift_range=0.2,
        rotation_range=15,
        shear_range=0.2,
    )
    
    v_datagen = ImageDataGenerator(
        featurewise_center= True,
        featurewise_std_normalization = True,
    )

    t_datagen.fit(X_train)
    v_datagen.fit(X_val)

    train_gen = t_datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen = v_datagen.flow(X_val, y_val, batch_size=32, shuffle=True)
    
    #number of iteration to run the training for
    epochs = 25
    
    #learning rate callback to adjust weights after x epochs, changes depending on model
    LR_scheduler = LearningRateScheduler(learnR_schedule)

    if name == 'resnet50':
        LR_scheduler = LearningRateScheduler(learnR_schedule_ResNet50)
        
    #patience will stop the algorithm after x epochs if validation loss does not decrease further
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1, callbacks=[early_stopping, LR_scheduler])

    #Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('accuracy.png')
    
    #clear previous plot
    plt.clf()

    #Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('loss.png')

    # save model
    model.save('skin_cancer_diagnosis_model.h5')


def Evaluate_Model():

    #load the model
    model = load_model('skin_cancer_diagnosis_model.h5')
    
    #load the test data
    X_test, y_test = load_test_data()

    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    #standarise images
    datagen = ImageDataGenerator(featurewise_center= True,
        featurewise_std_normalization = True)

    datagen.fit(X_test)

    test_generator = datagen.flow(X_test, y_test, batch_size=32, shuffle=False)    
    
    
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(test_generator, verbose=1)

    # Print the evaluation metrics
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    
    y_pred = model.predict(test_generator)
    y_pred_binary =  np.where(y_pred >= 0.5, 1, 0)
        
    print('\nConfusion Matrix:\n')    
    print(confusion_matrix(y_test, y_pred_binary));
    
    #accuracy: (true positives + true negatives) / (positives + negatives)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print('\nAccuracy: %f' % accuracy)
    
    #recall
    recall = recall_score(y_test, y_pred_binary)
    print('Recall: %f' % recall)

    #precision true positives / (true positives + false positives)
    precision = precision_score(y_test, y_pred_binary)
    print('Precision: %f' % precision)
    
    #f1
    f1 = f1_score(y_test, y_pred_binary)
    print('F1 score: %f' % f1)    
       
    #ROC AUC
    auc = roc_auc_score(y_test, y_pred_binary)
    print('ROC AUC: %f' % auc)
    
    
    #calculate roc curves
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    
        
    #plot the roc curve for the model
    plt.figure()
    plt.plot(fpr, tpr, linestyle='--', label='')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('ROC.png')

#Use curated model
#model = Create_Model()
#Train_Model(model, name='custommodel')
#Evaluate_Model()

#Use pretrained EfficientNetB3 model
#model = Create_EfficientNetB3_Model()
#Train_Model(model, name='efficientnetb3')
#Evaluate_Model()

#Use custom resnet50 model
model = Create_My_ResNet50_Model()
Train_Model(model, name='resnet50')
Evaluate_Model()
