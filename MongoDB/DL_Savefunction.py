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
            "Mkey": int(Mkey),
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

    """
       *************************************   查詢區域   ****************************************************
    """
    # 查詢"Train_List"中,特定Serial的值
    def Find_Train_List_Serial(self, serial):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Train_List')
        find_txt = {"serial": {"$eq": int(serial)}}
        Result = list(self.Find(find_txt, show_id=False))
        return Result

    # 查詢"Train_Parameter"中,特定Mkey(等同serial)的值
    def Find_Train_Parameter_Mkey(self, Mkey):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Train_Parameter')
        find_txt = {"Mkey": {"$eq": int(Mkey)}}
        Result = list(self.Find(find_txt, show_id=False))
        return Result

    # 查詢"Predict_List"中,特定PredKey的值
    def Find_Pred_List_Serial(self, PredKey):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Predict_List')
        find_txt = {"PredKey": {"$eq": int(PredKey)}}
        Result = list(self.Find(find_txt, show_id=False))
        return Result

    # 查詢已經Predict完成的案件
    def Find_Pred_List_Finish(self):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Predict_List')
        find_txt = {"Finish": {"$eq": True}}
        Result = list(self.Find(find_txt, show_id=False))
        return Result

    # 查詢指定Train的history
    def Find_Train_history(self, Serial_Mkey):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Train_History')
        find_txt = {"Mkey": {"$eq": Serial_Mkey}}
        Result = list(self.Find(find_txt, show_id=False))
        return Result

    """
       *************************************   輸入區域   ****************************************************
    """
    # 將訓練結束後History記錄下來至Train_History
    def Insert_Train_History(self, pb_history_Mkey):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Train_History')
        Result = self.Insert(pb_history_Mkey)
        print("Insert_Train_History:", Result)
    """
       *************************************   更新區域   ****************************************************
    """
    # 將訓練狀態修改成 Stop or Start
    def Trainning_Call_StopStart(self, Train_serial, call_STOP_status=True):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Train_List')
        Update_Con = {"serial": {"$eq": int(Train_serial)}}
        if call_STOP_status:
            Result = self.Update(Update_Con, {"Stop": True})
            print("Trainning_Call_Stop:", Result)
        else:
            Result = self.Update(Update_Con, {"Stop": False})
            print("Trainning_Call_Stop:", Result)

    # 將預測狀態修改成 Stop or Start
    def Predict_Call_StopStart(self, Train_serial, call_STOP_status=True):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Predict_List')
        Update_Con = {"Mkey": {"$eq": int(Train_serial)}}

        if call_STOP_status:
            Result = self.Update(Update_Con, {"Stop": True})
            print("Predict_Call_Stop:", Result)
        else:
            Result = self.Update(Update_Con, {"Stop": False})
            print("Predict_Call_Stop:", Result)

    # 登記訓練開始時間
    def Update_Trainning_Time(self, Train_serial, Type_Start=True):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Train_List')

        if Type_Start:
            Update_Con = {"serial": {"$eq": int(Train_serial)}}
            Result = self.Update(Update_Con, {"Strat_Date": datetime.now().strftime("%Y/%m/%d %H:%M:%S")})
            print("Trainning_start_time:", Result)
        else:
            Update_Con = {"serial": {"$eq": int(Train_serial)}}
            Result = self.Update(Update_Con, {"End_Date": datetime.now().strftime("%Y/%m/%d %H:%M:%S")})
            print("Trainning_start_time:", Result)

    # 登記預測開始時間
    def Update_Precict_Time(self, Pred_Mkey, Type_Start=True):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Predict_List')

        if Type_Start:
            Update_Con = {"Mkey": {"$eq": int(Pred_Mkey)}}
            Result = self.Update(Update_Con, {"Strat_Date": datetime.now().strftime("%Y/%m/%d %H:%M:%S")})
            print("Predict_start_time:", Result)
        else:
            Update_Con = {"Mkey": {"$eq": int(Pred_Mkey)}}
            Result = self.Update(Update_Con, {"End_Date": datetime.now().strftime("%Y/%m/%d %H:%M:%S")})
            print("Predict_start_time:", Result)

    # 變更訓練結束時的狀態
    def Update_Trainning_Finish(self, Train_serial):
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Train_List')

        Update_Con = {"serial": {"$eq": int(Train_serial)}}
        Result = self.Update(Update_Con, {"Finish": True,"Stop": False})

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

    def TrainList_History_Del(self, Train_serial):
        #  刪除掉訓練完成的History
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Train_History')
        del_txt = {"Mkey": {"$eq": int(Train_serial)}}

        Result = self.Delete(del_txt)
        print("Tranning DELETE -History part", Result)

    def PredList_Del(self, Pred_serial):
        # 直接砍掉Pred那個選項
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Predict_List')
        del_txt = {"Predkey": {"$eq": int(Pred_serial)}}

        Result = self.Delete(del_txt)
        print("Predict DELETE - PredictList part", Result)

    def Pred_Result_Del(self, Pred_serial):
        # 直接砍掉Pred那個選項
        self.ConnDatabase('FlaskWeb')
        self.ConnCollection('Predict_Result')
        del_txt = {"Predkey": {"$eq": int(Pred_serial)}}

        Result = self.Delete(del_txt)
        print("Predict DELETE - PredictList part", Result)


if __name__ == "__main__":
    uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
    spawnboo_MDB = MongoDB_Training(uri)
    rows = spawnboo_MDB.Find_Train_Parameter_Mkey(3)
    rows = rows[0].keys()
    for r in rows:
        print(r)


