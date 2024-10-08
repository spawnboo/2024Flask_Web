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
    import numpy as np

    img_size = (224, 224)
    img_shape = (img_size[0], img_size[1], 3)
    batch_size = 16
    train_data_path = r"D:\DL\chest_xray\train"
    val_data_path = r"D:\DL\chest_xray\val"
    test_data_path = r"D:\DL\chest_xray\test"

    # Train
    train_df = Data_Dataframe_process(train_data_path)
    print(train_df)
    # 產生一個 ImageDataGenerator 後續可以串接 影像前處理方法
    train_Datagen = ImageDataGenerator(preprocessing_function=scalar)
    # Takes the dataframe and the path to a directory + generates batches.
    # The generated batches contain augmented/normalized data.
    # 使用datafraom串接[當然有其他格式串接]  變成directory 和 genertory 形式 (前處理後的)
    train_gen = train_Datagen.flow_from_dataframe(dataframe=train_df,
                                                  x_col='filepaths',
                                                  y_col='label',
                                                  class_mode='binary',
                                                  target_size=img_size,
                                                  batch_size=batch_size,
                                                  shuffle=True)
    data_list = []
    batch_index = 0
    while batch_index <= train_gen.batch_index:
        data = train_gen.next()
        # print("data:",data)
        data_list.append(data[0])
        batch_index = batch_index + 1
    data_array = np.asarray(data_list)
    print("data_array:", data_array)


    # Valadation
    val_df = Data_Dataframe_process(val_data_path)
    # print(val_df)
    # 產生一個 ImageDataGenerator 後續可以串接 影像前處理方法
    val_Datagen = ImageDataGenerator(preprocessing_function=scalar)
    val_gen = val_Datagen.flow_from_dataframe(dataframe=val_df,
                                              x_col='filepaths',
                                              y_col='label',
                                              class_mode='binary',
                                              target_size=img_size,
                                              batch_size=batch_size,
                                              shuffle=False)

    # test
    test_df = Data_Dataframe_process(test_data_path)
    # print(test_df)
    # 產生一個 ImageDataGenerator 後續可以串接 影像前處理方法
    test_Datagen = ImageDataGenerator(preprocessing_function=scalar)
    test_gen = test_Datagen.flow_from_dataframe(dataframe=test_df,
                                              x_col='filepaths',
                                              y_col='label',
                                              class_mode='binary',
                                              target_size=img_size,
                                              batch_size=batch_size,
                                              shuffle=False)