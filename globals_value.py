# coding=utf-8
"""
    這邊是擺放整個全域變數的地方
"""
# 初始化方法, 一定要放在方法的最入口處
def global_initialze():
    global model_stop, train_project_serial, predict_project_serial

    # 停止的變數
    model_stop = False
    # 訓練中狀態的變數
    train_project_serial = ""
    # 預測中中狀態的變數
    predict_project_serial = ""
