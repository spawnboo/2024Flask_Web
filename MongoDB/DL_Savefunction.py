from MongoDB.MongoDB_Client import MDB
from datetime import datetime


class MongoDB_Training(MDB):
    # 查詢序列號碼, 並返回最大值+1
    def serialNUM(self, keyword, inc=1):
        insert_txt = { str(keyword) : { "$gt": -1 } }
        Serach = list(self.Find(insert_txt, sort=[(str(keyword), 1)], show_id=False))
        if len(Serach)==0:  # 當keyword沒有被建立過時, 直接建立0  * 有可能存在keyword 輸入錯誤
            insert_txt = {str(keyword): 0}
            self.Insert([insert_txt])
            return inc

        Targer = Serach[-1][str(keyword)]
        return Targer + inc

    # 產生訓練任務的紀錄[永久]
    def create_train_MainSQL(self, Mission_Name='', Creater='', Model='', serialKey='Mkey'):
        if Mission_Name == '':                          # 簡易防呆 填值
            Mission_Name = "Training_1"                 # 後面可以查詢任務值增加序列值
        if Creater == '':
            Creater = "Unknow"
        if Model == '':
            Model = "Unknow"
        # ==================================================================
        Mkey = self.serialNUM(serialKey)
        insert_dixt = {
            "serial": Mkey,
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

        return Mkey

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


if __name__ == "__main__":
    uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
    spawnboo_MDB = MongoDB_Training(uri)
    spawnboo_MDB.ConnDatabase('FlaskWeb')
    spawnboo_MDB.ConnCollection('coustom')

    a = spawnboo_MDB.serialNUM('Mkey')
    print(a)