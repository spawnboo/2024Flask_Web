from MongoDB_Client import MDB
from datetime import datetime


class MongoDB_Training(MDB):

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
    # uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
    # spawnboo_MDB = MongoDB_Training(uri)

    # 两个字典
    dict1 = {'a': 10, 'b': 8}
    dict2 = {'b': 6, 'c': 4}

    # 返回  None
    dict2.update(dict1)
    print(dict2)