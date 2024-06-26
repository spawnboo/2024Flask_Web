import FlaskWeb as FW
from MongoDB.MongoDB_Client import MDB
from base_Model.Spawn_model import spawnboo_model

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

def train():
# #MongoDB 產生物件
#     uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
#     mdb = MDB(uri)
#     # 選擇資料庫database
#     mdb.MDB_Database('FlaskWeb')
#     mdb.MDB_Collection('Train_history')

# 產生訓練的方法
    # 接收到的參數值 from flask
    img_size = (224, 224)
    img_shape = (img_size[0], img_size[1], 3)
    batch_size = 16
    train_data_path = r"D:\DL\chest_xray\train"

    # 從資料夾抓取資料變成DataFrame的方法
    train_df = Data_Dataframe_process(train_data_path)
    # [這邊需要改寫] 產生餵入資料的flow 後面需要變成class 然後 把各種前處理的選項加進去 給flask介面選擇
    train_Datagen = ImageDataGenerator(preprocessing_function=scalar)
    train_gen = train_Datagen.flow_from_dataframe(train_df,
                                                  x_col='filepaths',
                                                  y_col='label',
                                                  target_size=img_size,
                                                  class_mode='categorical',
                                                  color_mode='rgb',
                                                  shuffle=True,
                                                  batch_size=batch_size)


    # [全都要改寫] 產生要訓練的Model, 從flask選擇方法與各種參數後 變成一個model return
    # [改寫] 需要有callback的選項可以選, 何時停 紀錄甚麼參數?
    # ====================== 前置參數 ========================
    classes = list(train_gen.class_indices.keys())
    # =======================================================
    model = spawnboo_model(classes)
    model.EfficientNetB3_keras()

    # 這就是建好訓練模型的model 物件
    model.start_train(train_gen,Epochs=2)



if __name__ == "__main__":  # 如果以主程式運行
    # FW.Web_run()

    # trainthread = threading.Thread(target=train)
    # trainthread.start()
    # print("runing...")
    # time.sleep(5)
    #
    # f = open(r"C:\Users\Steven_Hsieh\Desktop\RR.txt", "w+")
    # f.writelines("False")
    # f.close()

    train()


# 上船記錄檔案到MongoDB
    # 寫入資料的方法
    # insert_txt = {"ip":"127.0.0.1"}
    # db_col = mdb.conn["FlaskWeb"]["coustom"].insert_one(insert_txt)

    # f = open(r"C:\Users\Steven_Hsieh\Desktop\RR.txt", "r")
    # print(f.read() == str(False))





