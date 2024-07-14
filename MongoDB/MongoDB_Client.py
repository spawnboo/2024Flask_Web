import certifi
from pymongo import MongoClient
from datetime import datetime


"""
這邊是設計一個Class 進入到一個Database
"""
class MDB:
    def __init__(self,uri):
        self.uri = uri
        self.Database = ''
        self.Collections = ''
        self.TableReady = False          # 檢查是否連上 DB 及 Collection

        # Create a new client and connect to the server  *新版本Mongo 需要有 certifi 產生憑證~ 連結雲端資料庫
        self.conn = MongoClient(uri, tlsCAFile=certifi.where())
        self.Connect_test() # 確保連線

    # 連線測試!
    def Connect_test(self):
        # Send a ping to confirm a successful connection
        try:
            self.conn.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

    # 檢查連線的Database名稱,並連線至Database, 同時清空self.collection
    def ConnDatabase(self, Databasename):
        # 查詢該資料庫中所有Database Name [防呆輸入錯誤]
        list_database_names =  self.conn.list_database_names()
        if Databasename in list_database_names:
            self.Database = Databasename
            self.Collections = ''

            self.TableReady = False
            print("Connect to Database '" + Databasename + "' ")
            return True
        else:
            self.Database = ''
            self.Collections = ''
            print("Database name '" + str(Databasename) + "' name not exist!!!")
            return False

    # 選擇目前Database中哪一個Collection
    def ConnCollection(self, CollectionName):
        if self.Database == '':
            print ("Please chooise your database first! now Database name = '" + self.Database + "' ")
        list_collection_names =  self.conn[self.Database].list_collection_names()
        if CollectionName in list_collection_names:
            print("Connect to Collection Name '" + CollectionName + "' ")
            self.Collections = CollectionName
            self.TableReady = True
            return True
        else:
            print("Collection name '" + str(CollectionName) + "' name not exist!!!")
            return False

    #============================================ CRUD 資料表查詢等功能 ==================================================
    # 查詢的方法
    def Find(self,conditionDict = {}, sort=[] , show_id = True):
        """
        :param conditionDict: {}  查詢的Dict
        :param show_id: bool    結果是否要帶入"_id"? [預設True]
        :return: 回傳查詢結果, 當查詢失敗時會回傳False
        """
        # 檢查Table狀況 防呆
        if not self.TableReady:
            print("Now Database = '" + self.Database + "' Collection = '" + self.Collections + " Check Please!!")
            return False

        if show_id:
            if sort != []:
                result = self.conn[str(self.Database)][str(self.Collections)].find(conditionDict, {}).sort(sort)
            else:
                result = self.conn[str(self.Database)][str(self.Collections)].find(conditionDict, {})
        else:
            if sort != []:
                result = self.conn[str(self.Database)][str(self.Collections)].find(conditionDict, {"_id": 0}).sort(sort)
            else:
                result = self.conn[str(self.Database)][str(self.Collections)].find(conditionDict, {"_id": 0})

        return result

    # 輸入的方法     *統一使用 instrt_many
    def Insert(self, Insert_dict):
        """
        :param Insert_dict: 以陣列型(list)式輸入[{inerst_data1},{inerst_data2}]
        :return: 回傳輸入結果的 ObjectID, 當查詢失敗時會回傳 False
        """
        # 檢查Table狀況 防呆
        if not self.TableReady:
            print("Now Database = '" + self.Database + "' Collection = '" + self.Collections + " Check Please!!")
            return False

        #CMD
        # self.conn[str(self.Database)][str(self.Collections)].insert_one(Insert_dict)
        result = self.conn[str(self.Database)][str(self.Collections)].insert_many(Insert_dict)
        return result

    # 更新方法  是使用update_many 請先確認是否皆是要修改
    def Update(self,conditionDict = {}, newValue = {}, emptyCondition = False):
        # 檢查Table狀況 防呆
        if not self.TableReady:
            print("Now Database = '" + self.Database + "' Collection = '" + self.Collections + " Check Please!!")
            return False

        # 防止空條件全部修改
        if not emptyCondition and conditionDict=={}:
            print("不可以空條件使用Update, 若確定要做請打開emptyCondition=True")
            return False

        # CMD
        Result = self.conn[str(self.Database)][str(self.Collections)].update_many(conditionDict, {"$set":newValue})
        return Result

    # 刪除方法  是使用deltet_many 請先確認是否皆是要刪除
    def Delete(self,conditionDict = {}, emptyCondition = False):
        # 檢查Table狀況 防呆
        if not self.TableReady:
            print("Now Database = '" + self.Database + "' Collection = '" + self.Collections + " Check Please!!")
            return False

        # 防止空條件全部修改
        if not emptyCondition and conditionDict=={}:
            print("不可以空條件使用Delete, 若確定要做請打開emptyCondition=True")
            return False

        # CMD
        Result = self.conn[str(self.Database)][str(self.Collections)].delete_many(conditionDict)
        return Result

if __name__ == "__main__":
    uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"

    mdb = MDB(uri)
    mdb.ConnDatabase('FlaskWeb')
    mdb.ConnCollection('Train_List')


    # dict = {'ip':'192.168.1.119'}
    # insert_txt = { "Mkey": 0 }
    # find_txt = { "Mkey": { "$gt": -1 } }
    # sort = [('G',1)]

    find_txt = {"$or": [
        {"Finish": {"$eq": False}},
        {"Stop": {"$eq": True}}]}


    # 查詢
    rows = mdb.Find(find_txt, show_id=False)
    for row in rows:
        print(row)
    # print(rows[0]["Finish"] == F)

    # # 輸入
    # Result = mdb.Insert([insert_txt])
    # print(Result)

    # # 修改
    # Update_Con = { "Mkey": { "$gt": -1 } }
    #
    # Result = mdb.Update(Update_Con, {"Batch_size":17})
    # print(Result)

    # # 刪除
    # Result = mdb.Delete(insert_txt)
    # print(Result.deleted_count)


