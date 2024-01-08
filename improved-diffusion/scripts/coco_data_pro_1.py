import pickle
import os
from tqdm import tqdm


def main():
    train_out_path = '../datasets/coco/coco_train_text_img.pkl'
    with open(train_out_path,'rb') as fr:
        all_data = pickle.load(fr)
    captions_raw = all_data["captions"]
    len_past = len(captions_raw)
    print("len(past):",len_past)
    train_all_embedding = []
    for i in tqdm(range(len_past)):
        d = captions_raw[i]
        data_dict = dict()
        data_dict["raw"]=d["raw"]
        data_dict["img"]=d["img"]
        data_dict["label"]=1
        train_all_embedding.append(data_dict)
        data_dict0 =dict()
        data_dict0["raw"] = d["raw"]
        data_dict0["img"] = d["img0"]
        data_dict0["label"] = 0
        train_all_embedding.append(data_dict0)
    train_out_path1='../datasets/coco/coco_train_text_img1.pkl'
    with open(train_out_path1, 'wb') as f:
        pickle.dump({"captions": train_all_embedding}, f)
    print("%0d train embeddings saved " % len(train_all_embedding))


if __name__ =='__main__':
    main()
