import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping

import globals_value as globals

# 這兩行是,檢索GPU or 指定GPU作業方式! 針對不同狀況下顯卡,會需要切成不同的狀況!
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

"""
    回呼函式, 做以下幾件事情:
        1. 接收停止函數指令, 停止訓練~
        2. 回傳目前訓練進度 *尚未準備好往前拋數值的想法和方法
"""
# *******************先不帶入, 會跳出警示!?  但可以用樣子******************************
class Mycallback(Callback):
    """Callback that terminates training when flag=1 is encountered.
    """
    def on_epoch_begin(self, epoch, logs={}):
      self.epoch = epoch    # 取得現在epoch作業狀態~ 回傳訓練進程

    def on_batch_end(self, batch, logs=None):
        Total_batch = self.params.get('steps')  # 這個是 max batch

        # 全域函數 告知停止
        if globals.model_stop == True:
            self.model.stop_training = True

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


# ---------------------------------------------------------------------------------
# 主要訓練的模型Class
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
        self.predictResult = []

        # 夾帶參數
        self.train_name = ''

    # ===================================輸入參數方法===============================================================
    def EfficientNet_parameter_test(self, img_sizeW = 224,img_sizeH = 224,batch_size = 32, Epoch = 2, drop_rate = 0.4,learning_rate = 0.001, summary = True):
        if img_sizeW % 8 == 0 and img_sizeH % 8 == 0:
            self.img_size = (img_sizeW, img_sizeH)

        if batch_size > 0: self.batch_size = batch_size
        if Epoch > 0 :self.Epochs = Epoch
        if drop_rate > 0 and drop_rate < 1 : self.drop_rate = drop_rate
        if learning_rate < 1 : self.learning_rate = learning_rate
        if type(summary) == bool : self.summary = summary

    # 輸入方法但輸入的是Dict [撰寫到SQL辦法]
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
                                                                       pooling='max')
        # base_model.trainable = False    # 凍結上層網路, 加速訓練

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

        if self.summary:            # 是否呈現模型結構
            self.model.summary()
    # ===================================執行方法===============================================================
    def start_train(self, train_gen, Epochs = 0, valid_gen ='', load_weightPATH=r"CNN_save\Training_1.h5"):
        if self.model == '':
            print ("Spawnboo_model() train model not build yet!")    # 防呆 沒有輸入資料!
            return False

        if Epochs == 0: Epochs = self.Epochs

        if load_weightPATH != '':
            self.model.load_weights(load_weightPATH)

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


        if globals.model_stop == False: # 抓全域函數
            # 有訓練成功的話, 紀錄Model
            if self.train_name == '':
                self.model.save_weights(r"CNN_save\eff.h5", overwrite=True)
                print("hishere1")
            else:
                self.model.save_weights("CNN_save/" + self.train_name + ".h5" , overwrite=True)
                print(r"CNN_save/" + self.train_name + ".h5")
                print("hishere2")
            print("traing Finish")
            return True
        else:
            globals.model_stop = False
            print("traing call stop!! and reset global 'model_stop' value to False~ ")
            return False

    def start_predict(self, predict_gen, load_weightPATH = r"CNN_save\Training_1.h5"):
        if self.model == '':
            print ("Spawnboo_model() train model not build yet!")    # 防呆 沒有輸入資料!
            return False

        # 讀取權重檔案, 若沒有讀到 是不給預測的!!
        if load_weightPATH != '':
            self.model.load_weights(load_weightPATH)
        else:
            print("model Predict! 這邊有權重檔案沒有載入! 需要確認一下喔!")
            return False

        self.predictResult = self.model.predict(predict_gen)
        return True

if __name__ == "__main__":  # 如果以主程式運行
    c=1
    # 預測方法