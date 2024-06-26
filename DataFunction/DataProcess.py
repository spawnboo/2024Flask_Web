import os
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 針對影像資料夾與特定的分類方式 自動產生成 Dataframe格式(pandas)
def Data_Dataframe_process(Image_processPath):
    # 檢查資料夾是否存在
    if os.path.isdir(Image_processPath) != True:
        print("資料夾路徑: " + Image_processPath + " 不存在!!")
        return -1

    data_path = Image_processPath
    filepaths = []
    labels = []
    folds = os.listdir(data_path)

    for fold in folds:
        f_path = os.path.join(data_path, fold)
        filelists = os.listdir(f_path)

        for file in filelists:
            filepaths.append(os.path.join(f_path, file))
            labels.append(fold)

    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='label')
    df = pd.concat([Fseries, Lseries], axis=1)


    return df


# (預留)影像增強,後續給ImageDataGenerator 帶入!
def scalar(img):
    return img


if __name__ == "__main__":  # 如果以主程式運行
    img_size = (224, 224)
    img_shape = (img_size[0], img_size[1], 3)
    batch_size = 16
    train_data_path = r"D:\DL\chest_xray\train"


    train_df = Data_Dataframe_process(train_data_path)
    #print(train_df)


    # 產生一個 ImageDataGenerator 後續可以串接 影像前處理方法
    train_Datagen = ImageDataGenerator(preprocessing_function=scalar)

    # Takes the dataframe and the path to a directory + generates batches.
    # The generated batches contain augmented/normalized data.
    # 使用datafraom串接[當然有其他格式串接]  變成directory 和 genertory 形式 (前處理後的)
    train_gen = train_Datagen.flow_from_dataframe(train_df,
                                                  x_col='filepaths',
                                                  y_col='label',
                                                  target_size=img_size,
                                                  class_mode='categorical',
                                                  color_mode='rgb',
                                                  shuffle=True,
                                                  batch_size=batch_size)


