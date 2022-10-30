import json
import os
import random


def get_data_path(path):
    truth_data_paths = []
    fake_data_paths = []
    test_data_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            name = os.path.splitext(file)[0]
            if name.find('cover_for_cls') == 0:
                truth_data_paths.append(os.path.join(root, file))
            elif name.find('result_cls') == 0:
                fake_data_paths.append(os.path.join(root, file))
            elif os.path.splitext(file)[1] !='.txt' and os.path.splitext(file)[1] !='.ckpt' :
                test_data_paths.append(os.path.join(root, file))

    return truth_data_paths, fake_data_paths, test_data_paths


def write_file(path, texts):
    with open(path, 'w', encoding='utf-8') as fw:
        for text in texts:
            fw.write(text)
        fw.close()


def get_data(truth_data_paths, fake_data_paths, train_path, dev_path, test_path):
    train = []
    dev = []
    test = []
    for path in truth_data_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data_list = f.readlines()
            text = [data.strip().replace('\n', '').replace('\r', '') + '\t1\n' for data in data_list]
            f.close()
        truth_length = len(text)
        train.extend(text[0:int(truth_length * 0.7)])
        dev.extend(text[int(truth_length * 0.7):int(truth_length * 0.9)])
        test.extend(text[int(truth_length * 0.9):truth_length])
    for path in fake_data_paths:
        print(path)
        with open(path, 'r', encoding='utf-8') as f:
            line = json.load(f)
            data_list = line['covertexts']
            text = [data.strip().replace('\n', '').replace('\r', '') + '\t0\n' for data in data_list]
            f.close()
        fake_length = len(text)
        train.extend(text[0:int(fake_length * 0.7)])
        dev.extend(text[int(fake_length * 0.7):int(fake_length * 0.9)])
        test.extend(text[int(fake_length * 0.9):int(fake_length)])
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    write_file(train_path, train)
    write_file(dev_path, dev)
    write_file(test_path, test)

def get_test_data(test_data_paths):

    for path in test_data_paths:
        with open(path, 'r', encoding='utf-8') as f:
            line = json.load(f)
            data_list = line['covertexts']
            text = [data.strip().replace('\n', '').replace('\r', '') + '\t0\n' for data in data_list]
            f.close()
        # write_path= os.path.splitext(path)[0]+'.txt'
        write_path='./dataset/test_data/'+os.path.splitext(path)[0].split('/')[2]+'_'+os.path.splitext(path)[0].split('/')[3]+'.txt'
        print(write_path)
        write_file(write_path,text)


if __name__ == '__main__':
    root_path = './dataset'
    train_path = './dataset/train.txt'
    dev_path = './dataset/dev.txt'
    test_path = './dataset/test.txt'
    truth_data_paths, fake_data_paths, test_data_paths = get_data_path(root_path)
    # get_data(truth_data_paths, fake_data_paths, train_path, dev_path, test_path)
    get_test_data(test_data_paths)
