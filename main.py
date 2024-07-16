import FlaskWeb as FW
from MongoDB.MongoDB_Client import MDB
import base_Model.Spawn_model as spm

import threading
from DataFunction.DataProcess import Data_Dataframe_process, scalar
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time


# 啟動Flask
# 登入系統 [不急]
# 開啟CNN訓練視窗[不急]
# 選擇訓練的資料夾與model
# 開始訓練

# 訓練進度展示

# 訓練開始
# 訓練的圖像處理
# 訓練的model 產生
# 訓練的過程.h5 與 訓練的參數

# 紀錄model
# 紀錄訓練的過程狀態





if __name__ == "__main__":  # 如果以主程式運行
    train_PATH = r'D:\DL\chest_xray\train'

    # 從資料夾抓取資料變成DataFrame的方法
    train_df = Data_Dataframe_process(train_PATH)
    # [這邊需要改寫] 產生餵入資料的flow 後面需要變成class 然後 把各種前處理的選項加進去 給flask介面選擇
    train_Datagen = ImageDataGenerator(preprocessing_function=scalar)
    train_gen = train_Datagen.flow_from_dataframe(train_df,
                                                  x_col='filepaths',
                                                  y_col='label',
                                                  target_size=(224,224),
                                                  class_mode='categorical',
                                                  color_mode='rgb',
                                                  shuffle=True,
                                                  batch_size=16)
    # [全都要改寫] 產生要訓練的Model, 從flask選擇方法與各種參數後 變成一個model return
    # [改寫] 需要有callback的選項可以選, 何時停 紀錄甚麼參數?
    # ====================== 前置參數 ========================
    classes = len(list(train_gen.class_indices.keys()))
    # =======================================================

    # 載入模型  試算classes 數量

    train_model = spm.spawnboo_model(classes=classes)
    train_model.EfficientNet_parameter_test()
    train_model.EfficientNetB3_keras()

    train_model.start_train(train_gen)








