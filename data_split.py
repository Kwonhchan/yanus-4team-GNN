import pickle
from ncustom import NDataSplitter





def save_data_splitter(data_splitter):
    with open('data_splitter.pkl', 'wb') as f:
        pickle.dump(data_splitter, f)

def load_data_splitter():
    with open('data_splitter.pkl', 'rb') as f:
        data_splitter = pickle.load(f)
    return data_splitter

try:
    # 저장된 데이터 스플리터를 불러옵니다.
    data_splitter = load_data_splitter()
    print("Loaded data splitter from saved file.")
except (FileNotFoundError, IOError):
    print("Saved file for data splitter not found. Creating new instance.")
    # 데이터 스플리터 인스턴스를 새로 생성합니다.
    dataset_path = 'Dataset/최종합데이터.csv'
    data_splitter = NDataSplitter(dataset_path)

    # 데이터 스플리터 객체를 저장합니다.
    save_data_splitter(data_splitter)