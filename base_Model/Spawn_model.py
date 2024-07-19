import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Activation , Dense , Dropout , BatchNormalization , GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LearningRateScheduler

from tqdm.keras import TqdmCallback

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# 先不帶入, 會跳出警示!?  但可以用樣子
class Mycallback(Callback):
    """Callback that terminates training when flag=1 is encountered.
    """
    def on_epoch_begin(self, epoch, logs={}):
      self.epoch = epoch

    def on_batch_end(self, batch, logs=None):
        Total_batch = self.params.get('steps')  # 這個是 max batch

        # if self.epoch == 2:
        #   print (f"\nStopping at , Batch {batch}")
        #   self.model.stop_training = True

        # f = open(r"C:\Users\Steven_Hsieh\Desktop\RR.txt", "r")
        # if f.read() == str(False):
        #     self.model.stop_training = True

        # train status
        print ("Train Status: "  + str(batch) + " / " + str(Total_batch) +"  {:.2f}".format(batch/Total_batch*100) + " %")

    def on_epoch_end(self, epoch, logs=None):
        # 儲存現在model狀態
        c=1

class spawnboo_model():
    def __init__(self, classes):
        # 預先放入參數的地方
        self.Epochs = 10
        self.batch_size = 16
        self.drop_rate = 0.4
        self.learning_rate = 0.0001
        self.summary = True
        self.classes = classes

        self.img_zie = ()

        self.model = ''

        self.history = {}

    # ===================================輸入參數方法===============================================================
    def EfficientNet_parameter_test(self, img_sizeW = 224,img_sizeH = 224,batch_size = 32, Epoch = 2, drop_rate = 0.4,learning_rate = 0.001, summary = True):
        if img_sizeW % 8 == 0 and img_sizeH % 8 == 0:
            self.img_size = (img_sizeW, img_sizeH)

        if batch_size > 0: self.batch_size = batch_size
        if Epoch > 0 :self.Epochs = Epoch
        if drop_rate > 0 and drop_rate < 1 : self.drop_rate = drop_rate
        if learning_rate < 1 : self.learning_rate = learning_rate
        if type(summary) == bool : self.summary = summary

    # 輸入方法但輸入的是Dict
    def EfficientNet_parameter(self, parameter_dict):
        img_sizeW = int(parameter_dict['Image_SizeW'])
        img_sizeH = int(parameter_dict['Image_SizeH'])
        batch_size = parameter_dict['Batch_size']
        Epoch = int(parameter_dict['Epoch'])
        drop_rate = float(parameter_dict['Drop_rate'])
        learning_rate = float(parameter_dict['Learning_rate'])
        # summary = parameter_dict['Epoch']
        # =================================================================================================
        if img_sizeW % 8 == 0 and img_sizeH % 8 == 0:
            self.img_size = (img_sizeW, img_sizeH)

        if batch_size > 0: self.batch_size = batch_size
        if Epoch > 0 :self.Epochs = Epoch
        if drop_rate > 0 and drop_rate < 1 : self.drop_rate = drop_rate
        if learning_rate < 1 : self.learning_rate = learning_rate
        # if type(summary) == bool : self.summary = summary

    # ===================================模型方法===============================================================
    def EfficientNetB3_keras(self):
        img_size = self.img_size
        img_shape = (img_size[0], img_size[1], 3)

        base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,
                                                                       weights='imagenet',
                                                                       input_shape=img_shape,
                                                                       #pooling='max',
                                                                       drop_connect_rate=self.drop_rate)
        base_model.trainable = False    # 凍結上層網路, 加速訓練

        # Old方法
        self.model = Sequential([
            base_model,
            BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
            Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
                  bias_regularizer=regularizers.l1(0.006), activation='relu'),
            Dropout(rate=self.drop_rate, seed=75),
            Dense(self.classes, activation='softmax')
        ])
        self.model.compile(optimizer=Adamax(learning_rate=self.learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # self.model = Sequential([
        #                         base_model,
        #                         GlobalAveragePooling2D(),
        #                         BatchNormalization(),
        #                         Dropout(self.drop_rate),
        #                         Dense(1),
        #                         BatchNormalization(),
        #                         Activation("sigmoid")])
        # 紀錄訓練的方法, 透過metrics

        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="binary_acc"),
            tf.keras.metrics.AUC(name="AUC"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
        # self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
        #                     loss=BinaryCrossentropy())

        # 是否呈現
        if self.summary:
            self.model.summary()

    # ===================================執行方法===============================================================
    def start_train(self, train_gen, Epochs = 0, valid_gen ='', load_weight=''):
        if self.model == '':
            print ("Spawnboo_model() train model not build yet!")    # 防呆 沒有輸入資料!
            return False

        if Epochs == 0: Epochs = self.Epochs

        if load_weight != '':
            self.model.load_weights(r"CNN_save\eff.h5")

        callbacks = [
            #              ModelCheckpoint("model_at_epoch_{epoch}.h5"),
            #              ReduceLROnPlateau(monitor='val_loss',
            #                             patience=2,
            #                             verbose=1,
            #                             factor=0.07,
            #                             min_lr=1e-9),
            EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        ]


        if valid_gen == '':
            self.history = self.model.fit(x= train_gen, epochs = Epochs, verbose = 1, validation_steps = None,
                                          shuffle = False, callbacks=[Mycallback(), callbacks])
        else:
            self.history = self.model.fit(x=train_gen, epochs=Epochs, verbose=1, validation_data=valid_gen,
                                          validation_steps=None, shuffle=False, callbacks=[Mycallback(), callbacks])

        print(self.history)
        self.model.save_weights(r"CNN_save\eff.h5", overwrite=True)
        print("traing Finish")
        return True

    def start_validation(self, validat_gen):
        if self.model == '':
            print ("Spawnboo_model() train model not build yet!")    # 防呆 沒有輸入資料!
            return False

        predict = self.model.predict(validat_gen)
        print(predict)

if __name__ == "__main__":  # 如果以主程式運行
    c=1
    # 預測方法