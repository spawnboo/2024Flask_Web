from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask import Flask,  render_template, redirect, url_for
from flask import request, flash, session
import MongoDB.DL_Savefunction as MDB

import secrets



import threading
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


# 訓練排程機器人
def TrainQueeueRobot():

    # 在每次 訓練空閒後,重新排一次訓練資料庫的順序

    # 把import整理在這邊 不用家載太多
    import base_Model.Spawn_model  as Spawn_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from DataFunction.DataProcess import Data_Dataframe_process, scalar

    # 將Train_List中,Finish=False and Stop=True的讀出來  Train_Parameter值讀出來,
    MDB.ConnDatabase('FlaskWeb')
    MDB.ConnCollection('Train_List')
    find_txt = {"$or":[
                {"Finish": {"$eq": False}},
                {"Stop": {"$eq": True}}]}
    find_Result = MDB.Find(find_txt, show_id=False)
    find_Result = list(find_Result)


    # 將最高順位的訓練排程出去
    if len(find_Result) > 0:    # 如果有訓練清單待辦
        # 查找 訓練細節
        MDB.ConnDatabase('FlaskWeb')
        MDB.ConnCollection('Train_Parameter')
        find_txt = { "Mkey": { "$eq": find_Result[0]['serial'] } }
        parameter_Result = MDB.Find(find_txt, show_id=False)    # 理論上只會找到一組  但要處理可能有兩組的情況
        parameter_Result = list(parameter_Result)

        if find_Result[0]['Model'] == "effB3":
            # 訓練資料整理
            # 從資料夾抓取資料變成DataFrame的方法
            train_df = Data_Dataframe_process(parameter_Result[0]['trainPath'])
            # [這邊需要改寫] 產生餵入資料的flow 後面需要變成class 然後 把各種前處理的選項加進去 給flask介面選擇
            train_Datagen = ImageDataGenerator(preprocessing_function=scalar)
            train_gen = train_Datagen.flow_from_dataframe(train_df,
                                                          x_col='filepaths',
                                                          y_col='label',
                                                          target_size=(parameter_Result[0]['Image_SizeW'], parameter_Result[0]['Image_SizeH']),
                                                          class_mode='categorical',
                                                          color_mode='rgb',
                                                          shuffle=True,
                                                          batch_size=parameter_Result[0]['Batch_size'])
            # [全都要改寫] 產生要訓練的Model, 從flask選擇方法與各種參數後 變成一個model return
            # [改寫] 需要有callback的選項可以選, 何時停 紀錄甚麼參數?
            # ====================== 前置參數 ========================
            classes = len(list(train_gen.class_indices.keys()))
            # =======================================================

            # 載入模型  試算classes 數量
            train_model = Spawn_model.spawnboo_model(classes=classes)
            train_model.EfficientNet_parameter(parameter_Result[0])
            train_model.EfficientNetB3_keras()

            # 開始訓練
            train_result = train_model.start_train(train_gen)
            # 如果訓練成功, 標註SQL為已訓練
            # if train_result:


# ======================================================================================================================
@app.route("/")  # 函式的裝飾 ( Decorator )，以底下函式為基礎，提供附加的功能，這邊 "/" 代表根目錄
def home():
    # # 嘗試連線至MDB
    # MDB.MDB_Insert(insert_dict)

    # # 開啟訓練排程機器人
    # # 註記保留, 多線程啟用訓練方法
    # thread = threading.Thread(target=TrainQueeueRobot)
    # thread.daemon = True         # Daemonize
    # thread.start()

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
# 登入的初始畫面
@app.route("/member", methods=['GET'])
@login_required
def member():
    return render_template("Member.html")


# 登入到待訓練列隊頁面
@app.route("/trainList", methods=['GET', 'POST'])  # 函式的裝飾 ( Decorator )，以底下函式為基礎，提供附加的功能，這邊 "/" 代表根目錄
#@login_required
def trainList():
    if request.method == 'GET':
        # 資料庫撈取訓練列隊清單
        MDB.ConnDatabase('FlaskWeb')
        MDB.ConnCollection('Train_List')
        find_txt = {"$or": [
            {"Finish": {"$eq": False}},
            {"Stop": {"$eq": True}}]}
        # 查詢
        train_List_Result = MDB.Find(find_txt, show_id=False)
        train_List_Result = list(train_List_Result)
        # # 範本用
        # cur = table_sample
        # headers = table_sample[0].keys()
        # return render_template("TrainList.html", headers=list(headers), data=list(cur))

        if len(train_List_Result) > 0:
            headers = train_List_Result[0].keys()
            cur = train_List_Result
            return render_template("TrainList.html", headers=list(headers), data=list(cur))
        return render_template("TrainList.html")

    # if request.method == 'POST':
    #     button_clicked = request.form['Tool_btn_D']
    #     print(button_clicked)


    # # 轉跳到Home的html頁
    # return render_template("TrainList.html")


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

    return redirect(url_for('startTrain'))

# 按下StartTrain 名子Button 事件
@app.route('/startTrain', methods=['POST', 'GET'])  # 這邊'/startTrain' 是對照HTML中 <form> action=[要轉跳的地方] </form>
def startTrain():


    # 跳回訓練排程頁籤, 查看訓練排隊狀況與訓練狀況
    return redirect(url_for('TrainSucess')) #轉址方法 至/TrainSucess

# 提示使用者 訓練已經建立並開始, 這邊觸發SQL紀錄 + 轉跳至訓練中心頁籤
@app.route('/TrainSucess', methods=['GET'])  # 開始成功訓練頁籤
def TrainSucess():
    return render_template("Trainsucess.html")  # 轉跳至正在訓練的中心

# ===========================================   預測相關方法   ========================================================

@app.route('/PredictPage', methods=['GET', 'POST'])  # 進入預測中心頁面
def PredictPage():
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


    if btn_function == 'Look': # 表示按的是查看
        c=1

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
        print(trainList_serial)
        print(train_parameter_Result)
        # 將字元傳遞給PredictSet頁面,準備預測設定


        return render_template("PredictSet.html",train_List_Result=train_List_Result ,train_parameter_Result=train_parameter_Result)

    if btn_function == 'Del': # 表示按的是刪除
        c=1



    return render_template("PredictPage.html")  # 轉跳至預測中心頁面

# 以選擇想使用的模型,進入預測流程
@app.route('/PredictSet', methods=['GET', 'POST'])  # 進入預測中心頁面
def PredictSet():
    if request.method == 'GET':
        return render_template("PredictSet.html")

    # # 從html上取回內容
    # ProjectName = request.form['ProjectName']
    # trainPath = request.form['trainPath']  # 取得html中 name== 'trainPath' 的文字
    # WeightSelect = request.form['weight-select']  # ***尚未實作方法***  選擇歷史模型
    # modelSelect = request.form['model-select']  # 取得html中 name== 'model' 的文字
    # Image_SizeW = int(request.form['Image_SizeW'])
    # Image_SizeH = int(request.form['Image_SizeH'])
    # Epoch = int(request.form['Epoch'])
    # Batch_size = int(request.form['Batch_size'])
    # Drop_rate = float(request.form['Drop_rate'])
    # Learning_rate = float(request.form['Learning_rate'])
    #
    # # 記錄到資料庫
    # # 建立使用者訓練清單排程
    # MDB.ConnDatabase('FlaskWeb')
    # MDB.ConnCollection('Train_List')
    # Mkey = MDB.create_train_MainSQL(Mission_Name=ProjectName,
    #                                 Creater=current_user.id,
    #                                 Model=modelSelect,
    #                                 serialKey='serial')
    #
    # MDB.ConnDatabase('FlaskWeb')
    # MDB.ConnCollection('Train_Parameter')
    # # 建立訓練內容資料庫
    # MDB.create_train_ParameterSQL(Mkey,
    #                               trainPath,
    #                               modelSelect,
    #                               Image_SizeW,
    #                               Image_SizeH,
    #                               Epoch,
    #                               Batch_size,
    #                               Drop_rate,
    #                               Learning_rate,
    #                               serialKey='Pkey')
    #
    #     return redirect(url_for('startTrain'))

    return render_template("PredictPage.html")

# ====================================================  轉址的功能 ====================================================


if __name__ == "__main__":  # 如果以主程式運行
    # (暫時)登入Mongodb 路徑與方法
    uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
    MDB = MDB.MongoDB_Training(uri)

    app.run(debug=True)  # 啟動伺服器





