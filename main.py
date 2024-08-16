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
    train_PATH = r'D:\DL\chest_xray\val'

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

    # # train
    # train_model.start_train(train_gen,Epochs=20)
    # history = train_model.history

    # # save train History
    # import pandas as pd
    # hist_df = pd.DataFrame(history.history)
    # hist_csv_file = './trainHistoryDict/history.csv'
    # with open(hist_csv_file, mode='w') as f:
    #     hist_df.to_csv(f)


    # # 匯出 訓練過程
    # # Define needed variables
    # tr_acc = history.history['accuracy']
    # tr_loss = history.history['loss']
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

    # predict
    predict_Result = train_model.start_predict(train_gen)
    print(predict_Result)
    print(type(predict_Result))
    y_pred = (np.argmax(predict_Result, axis=1))




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
    save_img = r'.\CNN_save\CNN_Chartimg.jpg'
    plt.savefig(save_img)

    plt.show()










