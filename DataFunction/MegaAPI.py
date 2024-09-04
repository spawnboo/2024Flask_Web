# coding=utf-8
"""
    這是使用MEGA API的python 方法頁面! 協助上下傳 上下載處理
"""
from mega import Mega

# 取得現存容量
class FLASK_MEGA():
    def __init__(self, mega_email, mega_password):
        self.login(mega_email, mega_password)

    # 載入帳號
    def login(self, mega_email, mega_password):
        self.mega = Mega()
        self.MEGA_USER = self.mega.login(mega_email, mega_password)

    # 取得目前空間內物件
    def get_files(self):
        files = self.MEGA_USER.get_files()
        print("MEGA Files:",files)
        return files
    # 取得目前剩餘空間
    def get_quota(self):
        quota = self.MEGA_USER.get_quota()
        print("Total Space: ", quota, " MB")
        return quota
    # 查找特定資料夾或文件
    def find(self, folder_name):
        folder = self.MEGA_USER.find(folder_name)
        # # Excludes results which are in the Trash folder (i.e. deleted)
        # folder = self.MEGA_USER.find(folder_name, exclude_deleted=True)
        print("folder:",folder)
        if folder != None:
            return True
        else:
            return False



    # # 上傳"單一檔案"
    # def upload_file(self):



if __name__ == "__main__":  # 如果以主程式運行
    mega_email = r"e01646166@hotmail.com"
    mega_password = r"Ss0961006178"

    FM = FLASK_MEGA(mega_email, mega_password)

    # **************  上傳檔案  ********************
    # upload_path = r"D:\LDD"
    # R = MEGA_user.upload(upload_path)
    # print(R)

    print(FM.find('FLASK'))
