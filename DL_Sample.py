import os
import shutil
import itertools
import pathlib
from PIL import Image
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import plotly.express as px

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Activation , Dense , Dropout , BatchNormalization

from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers

# import warnings
# warnings.filterwarnings('ignore')


# ======================================================================================================================
train_data_path = r'D:\DL\chest_xray\train'
filepaths = []
labels = []
folds = os.listdir(train_data_path)

for fold in folds:
    f_path = os.path.join(train_data_path, fold)
    filelists = os.listdir(f_path)

    for file in filelists:
        filepaths.append(os.path.join(f_path, file))
        labels.append(fold)

Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='label')
df = pd.concat([Fseries, Lseries], axis=1)

test_data_path = r'D:\DL\chest_xray\test'

filepaths = []
labels = []
folds = os.listdir(test_data_path)

for fold in folds:
    f_path = os.path.join(test_data_path, fold)
    filelists = os.listdir(f_path)

    for file in filelists:
        filepaths.append(os.path.join(f_path, file))
        labels.append(fold)

Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='label')
test = pd.concat([Fseries, Lseries], axis=1)

valid_data_path = r'D:\DL\chest_xray\valid'

filepaths = []
labels = []
folds = os.listdir(test_data_path)

for fold in folds:
    f_path = os.path.join(test_data_path, fold)
    filelists = os.listdir(f_path)

    for file in filelists:
        filepaths.append(os.path.join(f_path, file))
        labels.append(fold)

Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='label')
valid = pd.concat([Fseries, Lseries], axis=1)


# ======================================================================================================================



train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle= True, random_state= 42)
valid_df, test_df= train_test_split(dummy_df, train_size= 0.6, shuffle= True, random_state= 42)

img_size = (224 ,224)
batch_size = 16
img_shape= (img_size[0], img_size[1], 3)

# print (df)

# ================================================ image ================================================
# 使用 keras.ImageDataGenerator

# 影像增強
def scalar(img):
    return img

tr_gen = ImageDataGenerator(preprocessing_function= scalar)
ts_gen = ImageDataGenerator(preprocessing_function= scalar)



# 這邊出來的型態為 <tensorflow.python.keras.preprocessing.image.DataFrameIterator object at 0x000001ADE8CE8DF0>
# 因為使用pandas 整理成  dataframe型式  所以使用 <flow_from_dataframe> 的參數輸入!
# 根據不同使用方式 當然可以選擇不同的輸入方式
train_gen = tr_gen.flow_from_dataframe(train_df , x_col = 'filepaths' , y_col = 'label' , target_size = img_size ,
                                      class_mode = 'categorical' , color_mode = 'rgb' , shuffle = True , batch_size =batch_size)
valid_gen = ts_gen.flow_from_dataframe(valid_df , x_col = 'filepaths' , y_col = 'label' , target_size = img_size ,
                                       class_mode = 'categorical',color_mode = 'rgb' , shuffle= True, batch_size = batch_size)
test_gen = ts_gen.flow_from_dataframe(test_df , x_col= 'filepaths' , y_col = 'label' , target_size = img_size ,
                                      class_mode = 'categorical' , color_mode= 'rgb' , shuffle = False , batch_size = batch_size)

# ------------------------------------------------------- image show part -------------------------------------------------------
gen_dict = train_gen.class_indices
classes = list(gen_dict.keys())
images , labels = next(train_gen)


# 這邊只是呈現圖片 知道整理出的樣子
# plt.figure(figsize= (20,20))
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     image = images[i] / 255
#     plt.imshow(image)
#     index = np.argmax(labels[i])
#     class_name = classes[index]
#     plt.title(class_name , color = 'blue' , fontsize= 12)
#     plt.axis('off')
# ------------------------------------------------------- image show part -------------------------------------------------------

# ================================================ model part ================================================
img_size = (224, 224)
img_shape = (img_size[0] , img_size[1] , 3)
num_class = len(classes)

base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top = False , weights = 'imagenet',input_shape = img_shape, pooling= 'max')
model = Sequential([
    base_model,
    BatchNormalization(axis= -1 , momentum= 0.99 , epsilon= 0.001),
    Dense(256, kernel_regularizer = regularizers.l2(l= 0.016) , activity_regularizer = regularizers.l1(0.006),bias_regularizer= regularizers.l1(0.006) , activation = 'relu'),
    Dropout(rate= 0.4 , seed = 75),
    Dense(num_class , activation = 'softmax')
])
model.compile(Adamax(learning_rate = 0.0001) , loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
# ================================================ model part ================================================

#model.load_weights('./Weight/checkpoint.weights.h5')


# 紀錄模型
checkpoint = ModelCheckpoint(filepath= './Weight/checkpoint.weights.h5',
                             monitor='val_accuracy',
                             mode='max',
                             save_best_only=False,
                             save_weights_only=True)



Epochs = 1
# 有紀錄版本
# history = model.fit(x= train_gen , epochs = Epochs, verbose = 1, validation_data= valid_gen,validation_steps = None , shuffle = False, callbacks=[checkpoint])
# 無紀錄版本
history = model.fit(x= train_gen , epochs = Epochs, verbose = 1, validation_data= valid_gen,validation_steps = None , shuffle = False, callbacks=[])

# TqdmCallback(verbose=2)

# 紀錄 訓練過程
with open('trainHistoryDict/trainHistoryDict.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi, protocol=pickle.HIGHEST_PROTOCOL)
# 打開pickle 存取的dict方法
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)


# list紀錄 [0.8214285969734192]
train_acc = history.history['accuracy']
train_loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]

index_acc = np.argmax(val_acc)
val_highest = val_acc[index_acc]

Epochs = [i+1 for i in range(len(train_acc))]

loss_label = f'Best epochs = {str(index_loss +1)}'
acc_label = f'Best epochs = {str(index_acc + 1)}'

print ("train_acc: ", train_acc)
print ("train_loss: ", train_loss)




#Training history

# plt.figure(figsize= (20,8))
# plt.style.use('fivethirtyeight')
#
# plt.subplot(1,2,1)
# plt.plot(Epochs , train_loss , 'r' , label = 'Training Loss')
# plt.plot(Epochs , val_loss , 'g' , label = 'Validation Loss')
# plt.scatter(index_loss + 1 , val_lowest , s = 150 , c = 'blue',label = loss_label)
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.subplot(1,2,2)
# plt.plot(Epochs , train_acc , 'r' , label = 'Training Accuracy')
# plt.plot(Epochs , val_acc , 'g' , label = 'Validation Accuracy')
# plt.scatter(index_acc + 1 , val_highest , s = 150 , c = 'blue',label = acc_label)
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.tight_layout
# plt.show()