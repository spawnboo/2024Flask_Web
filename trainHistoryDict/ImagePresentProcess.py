# coding=utf-8
"""
    這個程式主要是:
        處理訓練後的歷史資料(history)
        處理預測後的結果資料(Predict)
        輸出相關報告與圖像
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


# 將訓練過程 history.csv 變成 圖像
def Train_history_Present(history):
    #　將CSV轉換成可使用之dict

    # Define needed variables
    tr_acc = history['accuracy']
    tr_loss = history['loss']
    # val_acc = history.history['val_accuracy']
    # val_loss = history.history['val_loss']
    # index_loss = np.argmin(val_loss)
    # val_lowest = val_loss[index_loss]
    # index_acc = np.argmax(val_acc)
    # acc_highest = val_acc[index_acc]
    Epochs = [i + 1 for i in range(len(tr_acc))]
    # loss_label = f'best epoch= {str(index_loss + 1)}'
    # acc_label = f'best epoch= {str(index_acc + 1)}'

    # Plot training history
    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label='Training loss')
    # plt.plot(Epochs, val_loss, 'g', label='Validation loss')
    # plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
    # plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    # plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    return plt

# Predict 後輸出混沌矩陣方法圖片
def CNN_Predict_Present(predict_Result, data_gen):
    # predict 結果轉換成 結果
    y_pred = (np.argmax(predict_Result, axis=1))

    g_dict = data_gen.class_indices
    classes = list(g_dict.keys())

    # Confusion matrix
    cm = confusion_matrix(data_gen.classes, y_pred)

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

    return plt


# 將轉換出來的plt 存成特定影像檔案
def plt_saveIMG(plt, save_name='history', SAVE_TYPE='.png'):
    plt.savefig(save_name+SAVE_TYPE)

if __name__ == "__main__":  # 如果以主程式運行
    import pandas as pd

    # # 這邊是 透過 history.csv 產出做事
    # hist_csv_file = './history.csv'
    # historyCSV = pd.read_csv(hist_csv_file)
    # # print(historyCSV)
    # # print(historyCSV['loss'])
    # result_plt = Train_history_Present(historyCSV)
    # plt_saveIMG(result_plt,
    #             save_name='history')

    # =========================  輸入Pred結果做事  =================================




