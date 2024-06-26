from flask import Flask, jsonify, render_template, redirect, url_for
from flask import request
import MongoDB.MongoDB_Client as MDB
from datetime import datetime

# from DataFunction.DataProcess import Data_Dataframe_process, scalar
# from base_Model.Spawn_model import spawnboo_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 開始Flask方法
app = Flask(__name__)  # __name__ 為 python 內建的變數，他會儲存目前程式在哪個模組下執行
# ======================================================================================================================
# HOME
@app.route("/")  # 函式的裝飾 ( Decorator )，以底下函式為基礎，提供附加的功能，這邊 "/" 代表根目錄
def home():
    # # 嘗試連線至MDB
    # MDB.MDB_Insert(insert_dict)

    # 登入者的資料紀錄[coustom](使用者IP,時間)
    insert_dict = {"ip":str(request.remote_addr)}
    insert_dict['log_time'] = datetime.now()

    print(insert_dict)
    #
    # 轉跳到Home的html頁
    return render_template("trainingset.html", date=insert_dict)

@app.route("/table", methods=['GET'])  # 函式的裝飾 ( Decorator )，以底下函式為基礎，提供附加的功能，這邊 "/" 代表根目錄
def table():
    cur = MDB.MDB_find()
    headers = cur[0].keys()

    # 轉跳到Home的html頁
    return render_template("data.html", headers=list(headers), data=list(cur))


# 按下StartTrain 名子Button 事件
@app.route('/startTrain', methods=['POST', 'GET'])  # 這邊'/startTrain' 是對照HTML中 <form> action=[要轉跳的地方] </form>
def submit():
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

    return redirect(url_for('TrainSucess')) #轉址方法 至/TrainSucess

# 提示使用者 訓練已經建立並開始, 這邊觸發SQL紀錄 + 轉跳至訓練中心頁籤
@app.route('/TrainSucess', methods=['GET'])  # 開始成功訓練頁籤
def TrainSucess():
    return render_template("Trainsucess.html")  # 轉跳至正在訓練的中心


# ====================================================  轉址的功能 ====================================================
@app.route('/success/<action>/<name>')
def success(name, action):
    return '{} : Welcome {} ~ !!!'.format(action, name)

# 沒有使用的參數 不能夾帶,會出現Error
@app.route('/trainPathPost/<name>')
def trainPathSuccess(name):
    return 'Input Train Path is : {}'.format(name)


def Web_run(host = "0.0.0.0", port = 8888):
    app.run(host=host, port=port)  # 啟動伺服器


if __name__ == "__main__":  # 如果以主程式運行
    # (暫時)登入Mongodb 路徑與方法
    uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
    MDB = MDB.MDB(uri)
    MDB.MDB_Database('sample_mflix')
    MDB.MDB_Collection('users')

    app.run(host="0.0.0.0", port=8888)  # 啟動伺服器

