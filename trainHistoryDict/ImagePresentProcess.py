# coding=utf-8
"""
    這個程式主要是:
        處理訓練後的歷史資料(history)
        處理預測後的結果資料(Predict)
        輸出相關報告與圖像
"""
import matplotlib.pyplot as plt


# 將訓練過程 history.csv 變成 圖像
def Train_history_Present(history):
    # 匯出 訓練過程
    # Define needed variables
    tr_acc = history.history['accuracy']
    tr_loss = history.history['loss']
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
    plt.savefig(r'./trainHistoryDict/TrainHistory.png', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":  # 如果以主程式運行
    import pandas as pd

    hist_csv_file = './history.csv'

    historyCSV = pd.read_csv(hist_csv_file)

    Train_history_Present(historyCSV)
