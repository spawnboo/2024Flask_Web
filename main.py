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





