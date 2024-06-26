import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Activation , Dense , Dropout , BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback

from tqdm.keras import TqdmCallback

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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

        self.train_model = ''

        self.history = {}

    # ===================================輸入參數方法===============================================================
    def EfficientNet_parameter(self, img_sizeW = 224,img_sizeH = 224,batch_size = 16, Epoch = 2, drop_rate = 0.4,learning_rate = 0.001, summary = True):
        if img_sizeW % 8 == 0 and img_sizeH % 8 == 0:
            self.img_zie = (img_sizeW, img_sizeH)

        if batch_size > 0: self.batch_size = batch_size
        if Epoch > 0 :self.Epochs = Epoch
        if drop_rate > 0 and drop_rate < 1 : self.drop_rate = drop_rate
        if learning_rate < 1 : self.learning_rate = learning_rate
        if type(summary) == bool : self.summary = summary

    # ===================================模型方法===============================================================
    def EfficientNetB3_keras(self):
        img_size = (224, 224)
        img_shape = (img_size[0], img_size[1], 3)
        num_class = len(self.classes)

        base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet',
                                                                       input_shape=img_shape, pooling='max')
        self.train_model = Sequential([
            base_model,
            BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
            Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
                  bias_regularizer=regularizers.l1(0.006), activation='relu'),
            Dropout(rate=0.4, seed=75),
            Dense(num_class, activation='softmax')
        ])
        self.train_model.compile(Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        if self.summary:
            self.train_model.summary()

    # ===================================執行方法===============================================================
    def start_train(self, train_gen, Epochs = 0, valid_gen =''):
        if self.train_model == '': print ("Spawnboo_model() train model not build yet!")    # 防呆 沒有輸入資料!

        if Epochs == 0: Epochs = self.Epochs

        if valid_gen == '':
            self.history = self.train_model.fit(x= train_gen , epochs = Epochs, verbose = 1,validation_steps = None , shuffle = False, callbacks=[Mycallback()])
        else:
            self.history = self.train_model.fit(x=train_gen, epochs=Epochs, verbose=1, validation_data=valid_gen,
                                 validation_steps=None, shuffle=False, callbacks=[Mycallback()])

        print(self.history)
        return "traing Finish"