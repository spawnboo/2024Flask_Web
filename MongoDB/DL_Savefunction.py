from MongoDB.MongoDB_Client import MDB
from datetime import datetime


class MongoDB_Training(MDB):
    # 查詢序列號碼, 並返回最大值+1
    def serialNUM(self, keyword, inc=1):
        insert_txt = { str(keyword) : { "$gt": -1 } }
        Serach = list(self.Find(insert_txt, sort=[(str(keyword), 1)], show_id=False))
        print(Serach)
        if len(Serach)==0:  # 當keyword沒有被建立過時, 直接建立0  * 有可能存在keyword 輸入錯誤
            insert_txt = {str(keyword): 0}
            self.Insert([insert_txt])
            return inc

        Targer = Serach[-1][str(keyword)]
        print("Targer:",Targer)
        return (Targer + inc)

    # 產生訓練任務的紀錄[永久]
    def create_train_MainSQL(self, Mission_Name='', Creater='', Model='', BaseModel = '' , serialKey='serial'):
        if Mission_Name == '':                          # 簡易防呆 填值
            Mission_Name = "Training_1"                 # 後面可以查詢任務值增加序列值
        if Creater == '':
            Creater = "Unknow"
        if Model == '':
            Model = "Unknow"
        if BaseModel == '':
            BaseModel = "Base"
        # ==================================================================
        SerialNUM = self.serialNUM(serialKey)    # 自動撈取後生成

        insert_dixt = {
            "serial": SerialNUM,
            "Mission_Name": str(Mission_Name),
            "Creater": Creater,
            "Model":Model,
            "Create_Date": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "Strat_Date":"",
            "End_Date":"",
            "Finish":False,
            "Stop":False,
            "BaseModel":BaseModel,
        }
        self.Insert([insert_dixt])

        return SerialNUM

    # 訓練參數紀錄
    def create_train_ParameterSQL(self, Mkey, trainPath, model, Image_SizeW, Image_SizeH, Epoch, Batch_size, Drop_rate, Learning_rate,  serialKey='Pkey'):

        # ==================================================================
        Pkey = self.serialNUM(serialKey)
        insert_dixt = {
            "Mkey": Mkey,
            "Pkey": Pkey,
            "trainPath": trainPath,
            "Model":model,
            "Image_SizeW": Image_SizeW,
            "Image_SizeH":Image_SizeH,
            "Epoch":Epoch,
            "Batch_size":Batch_size,
            "Drop_rate":Drop_rate,
            "Learning_rate": Learning_rate
        }
        self.Insert([insert_dixt])

        return 0

    # 產生預測任務的紀錄
    def create_pred_MainSQL(self, Mkey, predictPath, Mission_Name='', Creater='', Model='', serialKey='Predkey'):
        if Mission_Name == '':                          # 簡易防呆 填值
            Mission_Name = "Predict_1"                 # 後面可以查詢任務值增加序列值
        if Creater == '':
            Creater = "Unknow"
        if Model == '':
            Model = "Unknow"
        # ==================================================================
        Predkey = self.serialNUM(serialKey)    # 自動撈取後生成

        insert_dixt = {
            "Mkey": Mkey,
            "Predkey": Predkey,
            "predictPath":str(predictPath),
            "Mission_Name": str(Mission_Name),
            "Creater": Creater,
            "Model":Model,
            "Create_Date": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "Strat_Date":"",
            "End_Date":"",
            "Finish":False,
            "Stop":False,
        }
        self.Insert([insert_dixt])

        return Predkey

    # 預測結果紀錄
    def create_pred_ResultSQL(self, Predkey, PredictPath, model, PredNum, ACC,  serialKey='Rkey'):
        Rkey = self.serialNUM(serialKey)
        insert_dixt = {
            "Predkey": Predkey,
            "Rkey": Rkey,
            "PredictPath": PredictPath,
            "Model":model,
            "PredNum": PredNum,
            "ACC":ACC,
        }
        self.Insert([insert_dixt])

        return 0

    # 訓練菜單排程
    def Training_list(self):
        basic_dict = {}
        # 輸入前準備


        # 選擇Database and Collection 準備存入資料庫!
        self.ConnDatabase("FlaskWeb")
        self.Collections("Train_Schedule")

        # 寫入排程資料庫
        # self.Insert()

    # 訓練菜單紀錄


    # 訓練過程狀態

    # 將訓練/預測狀態修改成 Stop
    def Trainning_Call_Stop(self, Train_serial):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Train_List')

        Update_Con = { "serial": { "$eq": int(Train_serial) } }

        Result = self.Update(Update_Con, {"Stop":True})
        print("Trainning_Call_Stop:",Result)

    """
       *************************************   刪除區域   ****************************************************
    """
    # 刪除Trainning 區域的資料, 並移動到刪除紀錄那邊
    def TrainList_Del(self, Train_serial):
        #  先砍parameter
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Train_Parameter')

        del_txt = { "Mkey": { "$eq": int(Train_serial) } }

        Result = self.Delete(del_txt)
        print("Tranning DELETE -parameter part", Result)

        # 再砍原本train List中的主要值
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Train_List')

        del_txt = { "serial": { "$eq": int(Train_serial) } }

        Result = self.Delete(del_txt)
        print("Tranning DELETE -trainning part", Result)






if __name__ == "__main__":
    uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
    spawnboo_MDB = MongoDB_Training(uri)
    spawnboo_MDB.ConnDatabase('FlaskWeb')
    spawnboo_MDB.ConnCollection('coustom')

    a = spawnboo_MDB.serialNUM('Mkey')
    print(a)