"""打開暫存檔案得方法"""
import pickle

# 打開pickle 存取的dict方法
def pickleoOener(pickle_files):
    with open(pickle_files, 'rb') as handle:
        dict = pickle.load(handle)

    return dict

if __name__ == "__main__":
    pickle_files = r'./trainHistoryDict.pickle'
    dict = pickleoOener(pickle_files)
    print(dict)
    print(type(dict))
    print(dict.keys())

    for key in dict.keys():
        print(key)
        print(dict[str(key)])

