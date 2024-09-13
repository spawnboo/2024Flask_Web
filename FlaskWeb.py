from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask import Flask,  render_template, redirect, url_for
from flask import request, flash, session
import MongoDB.DL_Savefunction as MDB
from  trainHistoryDict.ImagePresentProcess import *
import secrets
import threading
import pandas as pd
import os
# 暫時性加入time的import 後續移出
import time



from RegisterEmail import Register_Function

# 臨時 或暫時存取物件
users = {'Me': {'password': '123'}}

table_sample = [{'Name': 'Zara', 'Age': 7},
                {'Name': 'Alex', 'Age': 10},
                {'Name': 'GG', 'Age': 3},
                {'Name': 'AC', 'Age': 120},
                {'Name': 'Ben', 'Age': 40},
                {'Name': 'Am', 'Age': 20},
                {'Name': 'Ys', 'Age': 19},
                {'Name': 'QQ', 'Age': 6},
                {'Name': 'F', 'Age': 70},
                {'Name': 'Fung', 'Age': 80}]




# ======================================================================================================================
# 全域函數
# global target_project_serial    # 目前排程機器人使用的函數或方法


# 開始Flask方法
app = Flask(__name__)  # __name__ 為 python 內建的變數，他會儲存目前程式在哪個模組下執行

app.secret_key = secrets.token_hex(16)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.session_protection = "strong"
# login_manager.login_view = 'login'
login_manager.login_message = '登入出現問題,請重新登入!!'



# 繼承 UserMinin 物件
class Member(UserMixin):
    pass


# 訓練/預測排程機器人
def TrainQueeueRobot():
    # 把import整理在這邊 不用加載太多, 未來確定再移出
    import base_Model.Spawn_model  as Spawn_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from DataFunction.DataProcess import Data_Dataframe_process, scalar

    # 不斷重複執行
    while True:
        # 在每次訓練空閒後,重新排一次訓練資料庫的順序
        # 將Train_List中,Finish=False and Stop=True的讀出來  Train_Parameter值讀出來,
        MDB.ConnDatabase('FlaskWeb')
        MDB.ConnCollection('Train_List')
        find_txt = {"$and":[
                    {"Finish": {"$eq": False}},
                    {"Stop": {"$eq": False}}]}
        train_find_result = list(MDB.Find(find_txt, show_id=False))



        # 將Predict_List中, Finish= False and Stop=True的讀出來
        MDB.ConnDatabase('FlaskWeb')
        MDB.ConnCollection('Predict_List')
        find_txt = {"$and":[
                    {"Finish": {"$eq": False}},
                    {"Stop": {"$eq": False}}]}
        pred_find_result = list(MDB.Find(find_txt, show_id=False))
        print("pred_find_result:", pred_find_result)


        # ***這邊會修改作法只有一個GPU 且未分割使用量! 會優先給Predict_List優先使用~ *未實作,未有第二張顯卡***

        # (Pred 預測)將最高順位的排程出去
        if len(pred_find_result) > 0: # 如果有預測清單待辦
            serial_Mkey = pred_find_result[0]['Mkey']
            PredKey = pred_find_result[0]['Predkey']            # 當前預測的serial or Mkey
            MDB.Update_Precict_Time(PredKey, Type_Start=True)   # 紀錄當前開始預測時間
            # 查找 訓練細節
            parameter_result = MDB.Find_Train_Parameter_Mkey(serial_Mkey)

            if pred_find_result[0]['Model'] == "effB3":
                # 從資料夾抓取資料變成DataFrame的方法
                pred_df = Data_Dataframe_process(pred_find_result[0]['predictPath'])
                # [這邊需要改寫] 產生餵入資料的flow 後面需要變成class 然後 把各種前處理的選項加進去 給flask介面選擇
                pred_Datagen = ImageDataGenerator(preprocessing_function=scalar)
                pred_gen = pred_Datagen.flow_from_dataframe(pred_df,
                                                            x_col='filepaths',  # 這邊'filepaths'是指抓取dataframe中,抬頭名稱的那一排
                                                            y_col='label',      # 這邊'label'是指抓取dataframe中,抬頭名稱的那一排
                                                            target_size=(parameter_result[0]['Image_SizeW'],
                                                                         parameter_result[0]['Image_SizeH']),
                                                            class_mode='categorical',
                                                            color_mode='rgb',
                                                            shuffle=False,      # 預測時一定不要重新排列! 會影響後續資料合併正確性
                                                            batch_size=parameter_result[0]['Batch_size'])
                # [全都要改寫] 產生要訓練的Model, 從flask選擇方法與各種參數後 變成一個model return
                # [改寫] 需要有callback的選項可以選, 何時停 紀錄甚麼參數?
                # ====================== 前置參數 ========================
                classes = len(list(pred_gen.class_indices.keys()))
                # =======================================================
                # 載入模型  試算classes 數量
                pred_model = Spawn_model.spawnboo_model(classes=classes)
                pred_model.EfficientNet_parameter_test()
                pred_model.EfficientNetB3_keras()
                # 開始預測
                #這邊要找是用哪個model 去預測,所以要撈一下MKey 找model 名稱
                MDB.ConnDatabase('FlaskWeb')
                MDB.ConnCollection('Train_List')
                print("pred_find_result:", pred_find_result)
                find_txt = {"serial": {"$eq": pred_find_result[0]["Mkey"]}}
                modelname_find_result = list(MDB.Find(find_txt, show_id=False))

                # 變更全域函數 訓練中的參數 - 開始
                globals.predict_project_serial = PredKey
                # 預測方法開始, 載入預設定模型
                if len(modelname_find_result) > 0:
                    print("predict function 有成功帶入模型", os.path.join("CNN_save", modelname_find_result[0]["Mission_Name"]+".h5"))
                    predict_Result_Status = pred_model.start_predict(pred_gen,
                                                              load_weightPATH= os.path.join("CNN_save", modelname_find_result[0]["Mission_Name"]+".h5") )  # *這邊模型到時候要修改成自動帶入特定位置
                else:
                    print("predict function 失敗!!!!")
                    predict_Result_Status = pred_model.start_predict(pred_gen,
                                                          load_weightPATH = r"CNN_save\Training_1.h5")  #   *這邊模型到時候要修改成自動帶入特定位置
                # 變更全域函數 訓練中的參數 - 結束
                globals.predict_project_serial = ""

                if predict_Result_Status:  # Predict是否成功跑完
                    MDB.Update_Precict_Time(PredKey, Type_Start=False)  # 紀錄結束的預測時間
                    MDB.Update_Predict_Finish(PredKey)
# TODO: 這邊要改寫成, 把每一個預測結果輸出與對應的clsaa名稱 + 預測的PredKey 記錄到Mongo中"Predict_Result"中, 並且會跑圖有混沌矩陣可以參考
                    print("=========輸出預測結果與相關結果圖樣==============")
                    pred_label_list = pred_df['label'].tolist()                                        # 預測清單中,clsses
                    classes_list = list(pred_gen.class_indices.keys())                              # 此次訓練的類別清單 *注意若有缺少類別則會錯誤
                    x_pred = [classes_list.index(cls) for cls in pred_label_list]                      # 將原始答案輸出變成矩陣1D list, EX: [1, 1, 0, 1, 0]
                    y_pred = (np.argmax(pred_model.predictResult, axis=1))                          # 將預測輸出矩陣變成1D ndarry, EX: [1 1 0 1 0]

                    # 計算該次CNN Predict 比對正確率
                    pred_ans = [x_pred[i] == y_pred[i] for i in range(len(y_pred))]
                    acc = pred_ans.count(True) / len(pred_ans)                                      # 精確度acc
                    # 將此次預測結果存到Mongo資料庫中, 與更新預測列表中的精確度數值
                    MDB.Insert_Pred_Result(PredKey=PredKey, X_df=pred_df, Y=y_pred, classes_List=classes_list)
                    MDB.Update_Predict_acc(PredKey, acc)
                    # 輸出圖片
                    cnn_pred_plt = CNN_Predict_Present(pred_model.predictResult, pred_gen)          # 輸出混沌矩陣圖片,至 /static/images/pred_confusion_result.png
                    plt_saveIMG(cnn_pred_plt, save_name='./static/images/cnn_pred_result', SAVE_TYPE='.png')
        #*********************************************************************************************
        # (Train 訓練)將最高順位的排程出去
        if len(train_find_result) > 0:    # 如果有訓練清單待辦
            serial_Mkey = train_find_result[0]['serial']    # 主要要訓練的serial編號
            MDB.Update_Trainning_Time(serial_Mkey, Type_Start=True)              # 更新/紀錄開始訓練時間

            parameter_result = MDB.Find_Train_Parameter_Mkey(serial_Mkey)            # 查找訓練參數Paremater
            if train_find_result[0]['Model'] == "effB3":
                # 從資料夾抓取資料變成DataFrame的方法
                train_df = Data_Dataframe_process(parameter_result[0]['trainPath'])
                # [這邊需要改寫] 產生餵入資料的flow 後面需要變成class 然後 把各種前處理的選項加進去 給flask介面選擇
                train_Datagen = ImageDataGenerator(preprocessing_function=scalar)
                train_gen = train_Datagen.flow_from_dataframe(train_df,
                                                              x_col='filepaths',        # 這邊'filepaths'是指抓取dataframe中,抬頭名稱的那一排
                                                              y_col='label',            # 這邊'label'是指抓取dataframe中,抬頭名稱的那一排
                                                              target_size=(parameter_result[0]['Image_SizeW'], parameter_result[0]['Image_SizeH']),
                                                              class_mode='categorical', # 返回標籤的方式, categorical=> 2D numpy one-hot形式
                                                              color_mode='rgb',
                                                              shuffle=True,
                                                              batch_size=parameter_result[0]['Batch_size'])
                # [全都要改寫] 產生要訓練的Model, 從flask選擇方法與各種參數後 變成一個model return
                # [改寫] 需要有callback的選項可以選, 何時停 紀錄甚麼參數?
                # ====================== 前置參數 ========================
                classes = len(list(train_gen.class_indices.keys()))
                # =======================================================
                # 載入模型  試算classes 數量
                train_model = Spawn_model.spawnboo_model(classes=classes)
                train_model.EfficientNet_parameter(parameter_result[0])
                train_model.EfficientNetB3_keras()
                # 這邊確定model存檔名稱
                train_model.train_name = train_find_result[0]['Mission_Name']

                # 變更全域函數 訓練中的參數 - 開始
                globals.train_project_serial = parameter_result[0]['Mkey']
                # 開始訓練
                train_result = train_model.start_train(train_gen)
                # 變更全域函數 訓練中的參數 - 結束
                globals.train_project_serial = ""

                # 如果訓練成功, 標註SQL為已訓練,  並記錄訓練過程(History)
                if train_result:    # 如果訓練成功[不包含停止訓練]
                    MDB.Update_Trainning_Finish(serial_Mkey)    # 訓練狀態(Finish=True)完成, 狀態變更
                    MDB.Update_Trainning_Time(serial_Mkey, Type_Start=False)  # 紀錄結束訓練時間
                    #***************************  紀錄訓練History  ************************************
                    Train_history = train_model.history.history # 訓練完成的歷史紀錄
                    if len(Train_history)>0:  # 確保有資料輸入
                        # 將其轉換成每一個row to dict
                        pd_history = pd.DataFrame(Train_history)
                        pd_history = pd_history.to_dict(orient='records')    # 將資料轉成[{'loss': 5.86, 'accuracy': 1.0}, {'loss': 5.83, 'accuracy': 0.9375}]
                        # 並將其加入該次訓練之Mkey
                        pd_history_withMkey = [dict({'Mkey':serial_Mkey}, **item, ) for item in pd_history]
                        MDB.Insert_Train_History(pd_history_withMkey)   # 將訓練的history  記錄下來
        ####################################   While 迴圈尾端  ##########################################
        print("訓練機器人暫時沒找到要訓練/預測的項目,休息1分鐘~")
        time.sleep(60)  # 一分鐘後再看有無新的資料

# 監控是否中斷訓練/預測
def TrainStopListenRobot():
    """
    目前使用的方法,並不是最好的!?
    目前讓此方法去撈資料庫目前狀況,做出同樣的訓練清單列表!
    竟可能的同步訓練中狀態,如果停止的話,理論上會停到同一個function
    :return:
    """
    while True:
        # 如果要停止的值正在作業中,就呼叫全域變數"model_stop"便成為stop 停止訓練
        # Trainning
        if globals.train_project_serial != "":
            # 查詢是否停止作業
            MDB.ConnDatabase('FlaskWeb')
            MDB.ConnCollection('Train_List')
            find_txt = { "serial": { "$eq": globals.train_project_serial } }
            train_find_Result = list(MDB.Find(find_txt, show_id=False))
            if train_find_Result[0]['Stop'] == True:
                globals.model_stop = True
        # Predict
        if globals.predict_project_serial != "":
            # 查詢是否停止作業
            MDB.ConnDatabase('FlaskWeb')
            MDB.ConnCollection('Predict_List')
            find_txt = { "Predkey": { "$eq": globals.predict_project_serial } }
            pred_find_Result = list(MDB.Find(find_txt, show_id=False))
            if pred_find_Result[0]['Stop'] == True:
                globals.model_stop = True
        ####################################   While 迴圈尾端  ##########################################
        time.sleep(5)  # 每5秒檢查一次

# ======================================================================================================================
# =====================================  正常Flask Route 的進入區域 ======================================================
# ======================================================================================================================

@app.route("/")  # 函式的裝飾 ( Decorator )，以底下函式為基礎，提供附加的功能，這邊 "/" 代表根目錄
def home():
    """
        1. 開啟訓練排程機器人
        2. 轉跳至登入頁面
    :return: Login Page
    """
    # 註記保留, 多線程啟用訓練方法
    # 線程任務指派
    QueenRobot_thread = threading.Thread(target=TrainQueeueRobot)
    StopReader_thread = threading.Thread(target=TrainStopListenRobot)
    # 線程任務狀態設定
    QueenRobot_thread.daemon = True         # Daemonize
    StopReader_thread.daemon = True
    # 線程任務開始[含檢查機制,防止開多個]
    if QueenRobot_thread.is_alive() == False:
        QueenRobot_thread.start()
    if StopReader_thread.is_alive() == False:
        StopReader_thread.start()
    #  轉跳至 登入畫面  等未來有空再做登入畫面
    return redirect(url_for('login'))

# 登入使用者 with cookie
# 檢查是否是正確的使用者與密碼?  與request_loader 一起需要做
@login_manager.user_loader
def user_loader(user):
    # 1. Fetch against the database a user by `id`
    if user not in users:
        return  render_template("wrong.html")

    # 2. Create a new object of `User` class and return it.
    member = Member()
    member.id = user

    return member

# 登入使用者 without cookie
# # 檢查是否是正確的使用者與密碼? 與user_loader 一起需要做 未來
# @login_manager.request_loader
# def request_loader(request):
#     使用者 = request.form.get('user_id')
#     if 使用者 not in users:
#         return
#
#     member = Member()
#     member.id = 使用者
#     member.is_authenticated = 使用者.form['password'] == users[使用者]['password']
#     print("B member.is_authenticated" + member.is_authenticated)
#     return member

# 登入畫面
@app.route('/login', methods=['GET', 'POST'])
def login():
    # 更新使用者的資料庫應該在這邊, 再user_loader or request_loader 方法裡,會浪費太多資源!
    if request.method == 'GET':
        return render_template("LoginPage.html")

    key_id = request.form['user_id']
    if (key_id in users) and (request.form['password'] == users[key_id]['password']):
        member = Member()
        member.id = key_id
        login_user(member)
        flash(f'{key_id}！歡迎加入鱷魚的一份子！')
        return redirect(url_for('trainList'))
    else:
        flash('登入失敗了...')
        return render_template("LoginPage.html")
    return 'Bad login...'      #Crash Lode

# 註冊的方法 route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':     # 是get方法 清空session
        session['Register_user'] = ''
        session['Register_password']   = ''
        session['Register_email'] = ''
        return render_template("Register.html")

    key_id = request.form['user_id']
    if key_id in users:             # 確認使用者帳號使偶被使用
        flash("使用者已經被使用,請更換使用者名稱!")
        return render_template("Register.html")
    else:                               # 可以使用的帳號
        session['Register_user'] = request.form['user_id']
        session['Register_password']   = request.form['password']
        session['Register_email'] = request.form['email']
        return redirect(url_for('register_verify'))
    flash('註冊發生錯誤...')
    return 'bad register...'

# 信箱註冊驗證頁面
@app.route('/register_verify', methods=['GET', 'POST'])
def register_verify():
    if request.method == 'GET':
        session['varify_code'] = Register_Function(session['Register_email'])   # 透過給予的email寄送驗證信
        return render_template("RegisterCheck.html")

    key_varify = request.form['verify_code']
    if key_varify == session['varify_code']:            # 驗證成功的話加入使用者內
        users[session['Register_user']] = {'password':session['Register_password']}

        flash("註冊成功! 請重新登入")
        return redirect(url_for('login'))

    return 'bad register verify...'

# 登出
@app.route('/logout')
def logout():
    user = current_user.get_id()
    logout_user()
    flash(f'{user}！歡迎下次再來！')
    return redirect(url_for('login'))

@app.route("/aboutme", methods=['GET'])
def aboutme():
    return render_template("AboutMe.html")

# ========================================   這邊是登入後的方法了   =====================================================
# # 登入的初始畫面
# @app.route("/member", methods=['GET'])
# @login_required
# def member():
#     return render_template("Member.html")

# 登入到待訓練列隊頁面, 包含訓練及預測
@app.route("/trainList", methods=['GET', 'POST'])  # 函式的裝飾 ( Decorator )，以底下函式為基礎，提供附加的功能，這邊 "/" 代表根目錄
@login_required
def trainList():
    if request.method == 'GET':
        # 資料庫取得抬頭名稱
        train_List_heder = MDB.Find_Train_List_Header()
        train_Parameter_heder = MDB.Find_Train_Train_Parameter_Header()
        # 資料庫撈取訓練列隊清單
        train_waiting_result = MDB.Find_Train_List_WaitTrain()
        Pred_waiting_result = MDB.Find_Predict_List_WaitTrain()
        print("train_waiting_result:",train_waiting_result)
        print("Pred_waiting_result:", Pred_waiting_result)
        if len(train_waiting_result) > 0 or len(Pred_waiting_result)>0:
            if len(train_waiting_result) > 0: train_waiting_result = train_waiting_result
            if len(Pred_waiting_result) > 0: Pred_waiting_result = Pred_waiting_result

            # 現在系統正在跑的訓練或檢測(全域函數) train or predict
            now_run = ""
            if globals.train_project_serial != "":now_run = globals.train_project_serial
            if globals.predict_project_serial != "": now_run = globals.predict_project_serial
            return render_template("TrainList.html", train_header=train_List_heder, train_wait_list=train_waiting_result,
                                   pred_header = train_Parameter_heder, pred_wait_list=Pred_waiting_result,
                                   now_run=now_run)
        return render_template("TrainList.html")

    # 如果他點擊任意待訓練"TOOL"選項!
    trainList_serial, btn_function = request.form["Tool_btn"].split("_")  # 分析是哪一個model 與要做哪一件事
    print("TrainList Choose Serial:", trainList_serial)
    print("TrainList Choose Function:", btn_function)

    if btn_function == 'Look':  # 表示按的是查看目待訓練或正在訓練內容(參數)
        return redirect(url_for('LookTrainParameter', serial_num=trainList_serial))

    if btn_function == 'Stop':  # 表示按的是停止這個功能
        # 修改成Stop:True
        MDB.Trainning_Call_StopStart(trainList_serial, call_STOP_status=True)

    if btn_function == 'Start':  # 表示按的是重新開始這個功能
        # 修改成Stop:False
        MDB.Trainning_Call_StopStart(trainList_serial, call_STOP_status=False)

    if btn_function == 'Del':  # 表示按的是刪除
        # 先將該筆文件刪除後 移動至刪除區域
        MDB.TrainList_Del(trainList_serial)

    return redirect(url_for('trainList'))

# 設定訓練菜單
@login_required
@app.route('/trainSet', methods=['POST', 'GET'])  # 這邊'/startTrain' 是對照HTML中 <form> action=[要轉跳的地方] </form>
def trainSet():
    if request.method == 'GET':
        return render_template("trainingset.html")

    # 從html上取回內容
    ProjectName = request.form['ProjectName']
    trainPath = request.form['trainPath']  # 取得html中 name== 'trainPath' 的文字
    WeightSelect = request.form['weight-select']        # ***尚未實作方法***  選擇歷史模型
    modelSelect = request.form['model-select']  # 取得html中 name== 'model' 的文字
    Image_SizeW = int(request.form['Image_SizeW'])
    Image_SizeH = int(request.form['Image_SizeH'])
    Epoch = int(request.form['Epoch'])
    Batch_size = int(request.form['Batch_size'])
    Drop_rate = float(request.form['Drop_rate'])
    Learning_rate = float(request.form['Learning_rate'])

    # 記錄到資料庫
    # 建立使用者訓練清單排程
    MDB.ConnDatabase('FlaskWeb')
    MDB.ConnCollection('Train_List')
    Mkey = MDB.create_train_MainSQL(Mission_Name=ProjectName,
                                    Creater=current_user.id,
                                    Model=modelSelect,
                                    serialKey='serial')

    MDB.ConnDatabase('FlaskWeb')
    MDB.ConnCollection('Train_Parameter')
    # 建立訓練內容資料庫
    MDB.create_train_ParameterSQL(Mkey,
                                  trainPath,
                                  modelSelect,
                                  Image_SizeW,
                                  Image_SizeH,
                                  Epoch,
                                  Batch_size,
                                  Drop_rate,
                                  Learning_rate,
                                  serialKey='Pkey')


    return redirect(url_for('trainList'))

# 當按Look的方法的時候, 會將數據傳過來並可以觀看狀態
@app.route('/LookTrainParameter/<serial_num>', methods=['GET'])  # 這邊'/startTrain' 是對照HTML中 <form> action=[要轉跳的地方] </form>
def LookTrainParameter(serial_num):
    # 查找事哪一筆serial in 'Train_List' SQL
    look_main_result = MDB.Find_Train_List_Serial(int(serial_num))
    look_other_result = MDB.Find_Train_Parameter_Mkey(int(serial_num))

    main_cur = look_main_result[0]
    para_cur = look_other_result[0]

    # 查找訓練過程的紀錄(History) 如果已訓練完成 則有訓練參數可以參考
    if look_main_result[0]['Finish'] == True:
        history_result = MDB.Find_Train_history(int(serial_num))
        pd_history = pd.DataFrame(history_result)

        print("pd_history:",pd_history)
        # 製作圖片的方法, 可做成方法
        tr_acc = pd_history['accuracy']
        tr_loss = pd_history['loss']
        Epochs = [i + 1 for i in range(len(tr_acc))]
        # Plot training history
        plt.figure(figsize=(20, 8))
        plt.style.use('fivethirtyeight')

        plt.subplot(1, 2, 1)
        plt.plot(Epochs, tr_loss, 'r', label='Training loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout
        # 必須存到 /static/img裡面 這樣圖片才會有效
        plt.savefig(r'./static/TrainHistory.png', bbox_inches='tight')
        #plt.show()
        # 接下來, 要把佔存圖片秀到呈現去
        return render_template("TrainingLook.html", main_cur=main_cur, para_cur=para_cur, image_show=True)

    return render_template("TrainingLook.html", main_cur=main_cur, para_cur=para_cur, image_show=False)
# ===========================================   預測相關方法   ========================================================

@app.route('/PredictPage', methods=['GET', 'POST'])  # 進入預測中心頁面
def PredictPage():
    # 清除session暫存特定值
    session.pop('Mkey', None)

    if request.method == 'GET':
        # 資料庫撈取已訓練完成的清單
        MDB.ConnDatabase('FlaskWeb')
        MDB.ConnCollection('Train_List')
        find_txt = {"$or": [
            {"Finish": {"$eq": True}}]}
        # 查詢
        Predict_List_Result = MDB.Find(find_txt, show_id=False)
        Predict_List_Result = list(Predict_List_Result)

        if len(Predict_List_Result) > 0:
            headers = Predict_List_Result[0].keys()
            cur = Predict_List_Result
            return render_template("PredictPage.html", headers=list(headers), data=list(cur))
        return render_template("PredictPage.html")

    # 如果他點擊某一 完成訓練的model~則進入準備預測的選單!
    trainList_serial, btn_function = request.form["Tool_btn"].split("_")    # 分析是哪一個model 與要做哪一件事
    print("Choose Serial:",trainList_serial)
    print("Choose Function:", btn_function)

    if btn_function == 'Pred': # 表示按的是預測
        # 查詢該Serial 對應的值
        MDB.ConnDatabase('FlaskWeb')
        MDB.ConnCollection('Train_List')
        find_txt = {"serial": {"$eq":int(trainList_serial)}}
        # 查詢
        train_List_Result = MDB.Find(find_txt, show_id=False)
        train_List_Result = list(train_List_Result)

        MDB.ConnDatabase('FlaskWeb')
        MDB.ConnCollection('Train_Parameter')
        find_txt = {"Mkey": {"$eq":int(trainList_serial)}}
        # 查詢
        train_parameter_Result = MDB.Find(find_txt, show_id=False)
        train_parameter_Result = list(train_parameter_Result)

        # 將MainKey準備在Session中,準備使用
        session['Mkey'] = trainList_serial
        return render_template("PredictSet.html",train_List_Result=train_List_Result ,train_parameter_Result=train_parameter_Result)

    if btn_function == 'Look': # 表示按的是查看
        # 由於這邊顯示的是訓練完成可供訓練的模型, 所以轉跳至trainParameter
        return redirect(url_for('LookTrainParameter', serial_num=trainList_serial))

    if btn_function == 'Del': # 表示按的是刪除
        # 先將該筆文件刪除後 移動至刪除區域
        MDB.TrainList_Del(trainList_serial)
        # 要多刪除一個 history
        MDB.TrainList_History_Del(trainList_serial)
    return render_template("PredictPage.html")  # 轉跳至預測中心頁面

# 以選擇想使用的模型,進入預測流程
@app.route('/PredictSet', methods=['GET', 'POST'])  # 進入預測中心頁面
def PredictSet():
    if request.method == 'GET':
        return render_template("PredictSet.html")

    # 取得擷取到的資訊
    ProjectName = request.form['ProjectName']
    predictPath = request.form['predictPath']  # 取得html中 name== 'trainPath' 的文字
    modelName = request.form['model_name']  # 取得html中 name== 'model' 的文字
    # 從Session取得Mkey值
    if session.get('Mkey', None) != None:
        Mkey = session.get('Mkey', None)
    else: Mkey = None

    # 這邊將要預測的物件建立成資料庫排程
    # 建立使用者訓練清單排程
    MDB.ConnDatabase('FlaskWeb')
    MDB.ConnCollection('Predict_List')
    PredKey = MDB.create_pred_MainSQL(Mkey,
                                      predictPath,
                                      Mission_Name=ProjectName,
                                      Creater=current_user.id,
                                      Model=modelName,
                                      serialKey='Predkey')


    redirect(url_for('PredictPage'))

# 查看已經完成的Predict結果
@app.route('/PredictResult', methods=['GET', 'POST'])  # 進入預測中心頁面
def PredictResult():
    if request.method == 'GET':
        # 讀取SQL 載入已經Predict Finish=True的相關資料
        FinishPredict_List_Result = MDB.Find_Pred_List_Finish()

        if len(FinishPredict_List_Result) > 0:
            headers = FinishPredict_List_Result[0].keys()
            cur = FinishPredict_List_Result
            return render_template("PredictResult.html", headers=list(headers), data=list(cur))
        return render_template("PredictResult.html")

    # 這邊傳回的是PredKey! 和點集哪一個選項
    PredKey, btn_function = request.form["Tool_btn"].split("_")  # 分析是哪一個model 與要做哪一件事
    print("Choose Serial:", PredKey)
    print("Choose Function:", btn_function)

    if btn_function == 'Look':
        # 表示按的是查看預測完成的結果, 下方會顯示圖片與混沌矩陣
        return redirect(url_for('PredictResult', serial_num=PredKey))

    if btn_function == 'Del':  # 表示按的是刪除
        # 先找出該專案是否預測完成,若是預測完成!則需要多清理預測後的結果資料
        predlist_result = MDB.Find_Pred_List_Finish(PredKey)
        if predlist_result[0]['Finish']:
            # 刪除一個 Predict Result
            MDB.Pred_Result_Del(PredKey)
        # 刪除訓練的那個專案
        MDB.PredList_Del(PredKey)

    return render_template("PredictPage.html")  # 轉跳至預測中心頁面


# 查詢預測內容的方法
@app.route('/LookPredParameter/<PredKey>', methods=['GET'])  # 這邊'/startTrain' 是對照HTML中 <form> action=[要轉跳的地方] </form>
def LookPredParameter(PredKey):
    # 查找事哪一筆serial in 'Train_List' SQL
    look_main_result = MDB.Find_Train_List_Serial(int(PredKey))
    look_other_result = MDB.Find_Pred_List_Serial(int(PredKey))

    main_cur = look_main_result[0]
    para_cur = look_other_result[0]
    return render_template("PredLook.html", main_cur=main_cur, para_cur=para_cur)
# ====================================================  轉址的功能 ====================================================

if __name__ == "__main__":  # 如果以主程式運行
    # 全域函數宣告與初始化
    import globals_value as globals
    globals.global_initialze()

    # (暫時)登入Mongodb 路徑與方法
    uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
    MDB = MDB.MongoDB_Training(uri)

    app.run(debug=True)  # 啟動伺服器





