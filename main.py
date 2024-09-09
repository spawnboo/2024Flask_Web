import FlaskWeb as FW
from MongoDB.MongoDB_Client import MDB
import base_Model.Spawn_model as spm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
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
    import globals_value as globals
    globals.global_initialze()
    # 以上202408月新增  需要給訓練系統初始化全域函數


    train_PATH = r'D:\DL\chest_xray\test'

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
                                                  shuffle=False,
                                                  batch_size=32)
    # [全都要改寫] 產生要訓練的Model, 從flask選擇方法與各種參數後 變成一個model return
    # [改寫] 需要有callback的選項可以選, 何時停 紀錄甚麼參數?
    # ====================== 前置參數 ========================
    classes = len(list(train_gen.class_indices.keys()))
    # =======================================================

    # 載入模型  試算classes 數量

    train_model = spm.spawnboo_model(classes=classes)
    train_model.EfficientNet_parameter_test()
    train_model.EfficientNetB3_keras()


    # ###############################################################################################################


    # ===================訓練的方法===================
    #
    # # train
    # train_model.start_train(train_gen,Epochs=3)
    # history = train_model.history.history   # history的dict

    # *************************  轉格式的方法  ********************************
    # import pandas as pd
    # # 1.轉成 CSV
    # hist_df = pd.DataFrame(history)
    # hist_csv_file = './trainHistoryDict/history.csv'
    # with open(hist_csv_file, mode='w') as f:
    #     hist_df.to_csv(f)
    #
    # # 2.轉成每一個row都是dict的list
    # hist_df = pd.DataFrame(history)
    # history_dict_list = hist_df.to_dict(orient='records')
    #
    # print(history_dict_list)
    # # *************************  載入csv的方法  ********************************
    # import pandas as pd
    # csv_path = 'trainHistoryDict/history.csv'
    # history = pd.read_csv(csv_path)
    #
    # # *************************  SQL抓取history方法  ********************************
    # import MongoDB.DL_Savefunction as MDB
    #
    # # (暫時)登入Mongodb 路徑與方法
    # uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
    # MDB = MDB.MongoDB_Training(uri)
    #
    # r = MDB.Find_Train_history(8)
    # print(type(r))
    #
    # history=pd.DataFrame(r)
    #
    # # *************************  轉圖片的方法  ********************************
    # # Define needed variables
    # tr_acc = history['accuracy']
    # tr_loss = history['loss']
    # # val_acc = history.history['val_accuracy']
    # # val_loss = history.history['val_loss']
    # # index_loss = np.argmin(val_loss)
    # # val_lowest = val_loss[index_loss]
    # # index_acc = np.argmax(val_acc)
    # # acc_highest = val_acc[index_acc]
    # Epochs = [i + 1 for i in range(len(tr_acc))]
    # # loss_label = f'best epoch= {str(index_loss + 1)}'
    # # acc_label = f'best epoch= {str(index_acc + 1)}'
    #
    # # Plot training history
    # plt.figure(figsize=(20, 8))
    # plt.style.use('fivethirtyeight')
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(Epochs, tr_loss, 'r', label='Training loss')
    # # plt.plot(Epochs, val_loss, 'g', label='Validation loss')
    # # plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
    # # plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    # # plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    #
    # plt.tight_layout
    # plt.savefig(r'./trainHistoryDict/TrainHistory.png', bbox_inches='tight')
    #
    # plt.show()



    # ===================預測的方法===================
    train_df_list = train_df['label'].tolist()
    print("train_df:", train_df)

    classes_list = list(train_gen.class_indices.keys())
    #print("list(train_gen.class_indices.keys()):",classes_list)
    # 將列表中的文字轉換成 classes 中的次序
    print("*********************************")

    # predict
    predict_Result = train_model.start_predict(train_gen)
    print("train_model.predictResult:", train_model.predictResult)
    y_pred = (np.argmax(train_model.predictResult, axis=1))
    # print(y_pred)

    x_pred = [classes_list.index(data) for data in train_df_list]
    ans = [x_pred[i] == y_pred[i] for i in range(len(y_pred))]
    acc = ans.count(True) / len(ans)
    print(ans)
    print(acc)
    print(round(acc, 2))

    # 匯出 混沌矩陣方法
    g_dict = train_gen.class_indices
    classes = list(g_dict.keys())

    # Confusion matrix
    cm = confusion_matrix(train_gen.classes, y_pred)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 將混沌矩陣儲存
    save_img = r'./static/images/cnn_pred_result.png'
    plt.savefig(save_img)

    plt.show()










