from MongoDB_Client import MDB
from datetime import datetime


class MongoDB_Training(MDB):
    # 查詢序列號碼, 並返回最大值+1
    def serialNUM(self, keyword, inc=1):
        insert_txt = { str(keyword) : { "$gt": -1 } }
        Targer = list(self.Find(insert_txt, sort=[(str(keyword),1)], show_id=False))[-1][str(keyword)]
        return Targer + inc

    # 產生訓練任務的紀錄[永久]
    def create_train_mission(self, Mission_Name='', Creater=''):
        # 簡易防呆 填值
        if Mission_Name == '':
            # 後面可以查詢任務值增加序列值
            Mission_Name = "Training_1"
        if Creater == '':
            Creater = "Unknow"
        # ==================================================================
        insert_dixt = {
            "serial": self.AutoSerialNUM("serial"),
            "Mission_Name": str(Mission_Name),
            "Creater": Creater,
            "Create_Date": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        }
        self.MDB_Insert(insert_dixt)

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