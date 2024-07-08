import secrets
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask import Flask,  render_template, redirect, url_for
from flask import request, flash, session
import MongoDB.MongoDB_Client as MDB
from datetime import datetime

from RegisterEmail import Register_Function
# from DataFunction.DataProcess import Data_Dataframe_process, scalar
# from base_Model.Spawn_model import spawnboo_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator



# 開始Flask方法
app = Flask(__name__)  # __name__ 為 python 內建的變數，他會儲存目前程式在哪個模組下執行

app.secret_key = secrets.token_hex(16)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.session_protection = "strong"
# login_manager.login_view = 'login'
login_manager.login_message = '登入出現問題,請重新登入!!'

users = {'Me': {'password': '123'}}

table_sample = [{'Name': 'Zara', 'Age': 7},
                {'Name': 'Alex', 'Age': 10},
                {'Name': 'Fung', 'Age': 80}]

# 繼承 UserMinin 物件
class Member(UserMixin):
    pass

# ======================================================================================================================
@app.route("/")  # 函式的裝飾 ( Decorator )，以底下函式為基礎，提供附加的功能，這邊 "/" 代表根目錄
def home():
    # # 嘗試連線至MDB
    # MDB.MDB_Insert(insert_dict)

    #  轉跳至 登入畫面  等未來有空再做登入畫面
    #return redirect(url_for('login'))

    return redirect(url_for('trainList'))

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
        return redirect(url_for('member'))
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
@app.route("/trainList", methods=['GET'])  # 函式的裝飾 ( Decorator )，以底下函式為基礎，提供附加的功能，這邊 "/" 代表根目錄
#@login_required
def trainList():
    if request.method == 'GET':
        # 資料庫撈取訓練列隊清單
        # cur = MDB.MDB_find()
        # headers = cur[0].keys()
        # 範本用
        cur = table_sample
        headers = table_sample[0].keys()
        return render_template("TrainList.html", headers=list(headers), data=list(cur))

    # 轉跳到Home的html頁
    return render_template("TrainList.html")


# 設定訓練菜單
@app.route('/TrainSet', methods=['POST', 'GET'])  # 這邊'/startTrain' 是對照HTML中 <form> action=[要轉跳的地方] </form>
def trainSet():
    return render_template("trainingset.html")

# 按下StartTrain 名子Button 事件
@app.route('/startTrain', methods=['POST', 'GET'])  # 這邊'/startTrain' 是對照HTML中 <form> action=[要轉跳的地方] </form>
def startTrain():
    trainPath = request.form['trainPath']  # 取得html中 name== 'trainPath' 的文字
    modelSelect = request.form['model']  # 取得html中 name== 'model' 的文字
    Image_SizeW = request.form['Image_SizeW']
    Image_SizeH = request.form['Image_SizeH']
    Epoch = request.form['Epoch']
    Batch_size = request.form['Batch_size']
    Drop_rate = request.form['Drop_rate']
    Learning_rate = request.form['Learning_rate']

    # if modelSelect == "effB3":
    #     # 產生訓練的方法
    #     # 接收到的參數值 from flask
    #     img_size = (int(Image_SizeW), int(Image_SizeH))
    #     batch_size = int(Batch_size)
    #     train_data_path = trainPath
    #
    #     # 從資料夾抓取資料變成DataFrame的方法
    #     train_df = Data_Dataframe_process(train_data_path)
    #     # [這邊需要改寫] 產生餵入資料的flow 後面需要變成class 然後 把各種前處理的選項加進去 給flask介面選擇
    #     train_Datagen = ImageDataGenerator(preprocessing_function=scalar)
    #     train_gen = train_Datagen.flow_from_dataframe(train_df,
    #                                                   x_col='filepaths',
    #                                                   y_col='label',
    #                                                   target_size=img_size,
    #                                                   class_mode='categorical',
    #                                                   color_mode='rgb',
    #                                                   shuffle=True,
    #                                                   batch_size=batch_size)
    #     # [全都要改寫] 產生要訓練的Model, 從flask選擇方法與各種參數後 變成一個model return
    #     # [改寫] 需要有callback的選項可以選, 何時停 紀錄甚麼參數?
    #     # ====================== 前置參數 ========================
    #     classes = list(train_gen.class_indices.keys())
    #     # =======================================================
    #     model = spawnboo_model(classes)
    #     model.EfficientNetB3_keras()
    #
    #     # 這就是建好訓練模型的model 物件
    #     model.start_train(train_gen, Epochs=int(Epoch))
    #
    #     return redirect(url_for('TrainSucess', modelname=modelSelect))

    # 跳回訓練排程頁籤, 查看訓練排隊狀況與訓練狀況
    return redirect(url_for('TrainSucess')) #轉址方法 至/TrainSucess

# 提示使用者 訓練已經建立並開始, 這邊觸發SQL紀錄 + 轉跳至訓練中心頁籤
@app.route('/TrainSucess', methods=['GET'])  # 開始成功訓練頁籤
def TrainSucess():
    return render_template("Trainsucess.html")  # 轉跳至正在訓練的中心


# ====================================================  轉址的功能 ====================================================


if __name__ == "__main__":  # 如果以主程式運行
    # (暫時)登入Mongodb 路徑與方法
    # uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
    # MDB = MDB.MDB(uri)

    app.run(debug=True)  # 啟動伺服器





