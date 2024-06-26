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




if __name__ == "__main__":
    uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
    mdb = MongoDB_Training(uri)
    mdb.MDB_Database('FlaskWeb')
    mdb.MDB_Collection('coustom')

    mdb.create_train_mission()