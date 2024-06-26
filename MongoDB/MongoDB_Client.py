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

        # Create a new client and connect to the server
        self.conn = MongoClient(uri)
        self.MDB_connect_test() # 確保連線

    # 連線測試!
    def MDB_connect_test(self):
        # Send a ping to confirm a successful connection
        try:
            self.conn.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

    # 連線至Database, 同時清空self.collection
    def MDB_Database(self, Databasename):
        list_database_names =  self.conn.list_database_names()
        if Databasename in list_database_names:
            self.Database = Databasename
            self.Collections = ''
            print("Connect to Database '" + Databasename + "' ")
        else:
            print("Database name '" + str(Databasename) + "' name not exist!!!")

    # 選擇哪一個Collection
    def MDB_Collection(self, CollectionName):
        if self.Database == '':
            print ("Please chooise your database first! now Database name = '" + self.Database + "' ")
        list_collection_names =  self.conn[self.Database].list_collection_names()
        if CollectionName in list_collection_names:
            print("Connect to Collection Name '" + CollectionName + "' ")
            self.Collections = CollectionName
        else:
            print("Collection name '" + str(CollectionName) + "' name not exist!!!")

    def MDB_find(self):
        result = self.conn[str(self.Database)][str(self.Collections)].find({},{"_id":0})
        return  result

    def MDB_Insert(self, Insert_dict):
        # 檢查 Database 與 Collection 存在
        print(self.Database, self.Collections)
        if self.Database == '' or self.Collections == '':
            print("Now Database = '" + self.Database + "' Collection = '" + self.Collections + "' , Pleace choose First!")

        # insert_txt = {"ip": "127.0.0.1"}

        #CMD
        self.conn[str(self.Database)][str(self.Collections)].insert_one(Insert_dict)

if __name__ == "__main__":
    uri = "mongodb+srv://e01646166:Ee0961006178@spawnboo.dzmdzto.mongodb.net/?retryWrites=true&w=majority&appName=spawnboo"
    mdb = MDB(uri)
    # mdb.MDB_Database('FlaskWeb')
    # mdb.MDB_Collection('coustom')

    mdb.MDB_Database('sample_mflix')
    mdb.MDB_Collection('users')

    r = mdb.MDB_find()


    headers = r[0].keys()
    for g in r:
        for i in g.values():
            print(i)
        print("===")

    # # str(datetime.now())
    # insert_txt = {"ip":"127.0.0.1"}
    # db_col = mdb.conn["FlaskWeb"]["coustom"].insert_one(insert_txt)


