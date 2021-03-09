'''     

'''
import numpy as np
import keras.backend as K
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
#np.random.seed(1)
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def select_random(model, f_list, x_test, y_test, delta, iterate):
    batch = delta
    arr = np.random.permutation(len(f_list))
    min_index0 = arr[0:30]
    acc_list = []
    for i in range(len(model)):
        acc_list.append([])
    for i in range(iterate):
        # random
        arr = np.random.permutation(len(f_list))
        start = int(np.random.uniform(0, len(f_list) - batch))
        min_index = arr[start:start + batch]
        min_index0 = np.append(min_index0, min_index)
        for j in range(len(model)):
            label = y_test[np.array(f_list)[min_index0]]
            orig_sample = x_test[np.array(f_list)[min_index0]]
            orig_sample = orig_sample.reshape(-1, 28, 28, 1)
            pred = np.argmax(model[j].predict(orig_sample), axis=1)
            acc = np.sum(pred == label) / orig_sample.shape[0]
            acc_list[j].append(acc)
            print("numuber of samples is {!s}, SDS acc is {!s}".format(
                orig_sample.shape[0], acc))
    return acc_list


def mode_count(mode_list, mode):
    count = 0
    for i in range(len(mode_list)):
        if mode_list[i] == mode:
            count += 1
    return count


def label_read(model, x_test, y_test):
    label_list = []
    for i in range(len(model)):
        print('model{} begins testing.'.format(i))
        label_list.append([])
        for j in range(len(y_test)):
            print('sample{} finishes testing.'.format(j))
            orig_sample = x_test[j]
            orig_sample = orig_sample.reshape(-1, 28, 28, 1)
            pred = np.argmax(model[i].predict(orig_sample), axis=1)
            label_list[i].append(pred[0])
    label_list.append([])
    for i in range(len(y_test)):
        label = y_test[i]
        label_list[-1].append(label)
    print('y_test gets.')
    dataframe = pd.DataFrame(label_list)
    dataframe.to_csv(r"test.csv")
    return label_list


def label_compare(model, label_list, y_test):
    from scipy import stats
    mode_list = []
    mode_counts = []
    con_list = []
    for i in range(len(y_test)):
        mode_store = []
        for j in range(len(model)):
            mode_store.append(label_list[j][i])
        mode = stats.mode(mode_store)[0][0]
        mode_list.append(mode)
        m_count = mode_count(mode_store, mode)
        mode_counts.append(m_count)
        print('mode of sample{} gets.'.format(i))
    compare_flag = []
    count = 0
    for i in range(len(y_test)):
        if mode_list[i] == y_test[i]:
            a = 1
            compare_flag.append(a)
            count = count + 1
            con_list.append(i)
        else:
            a = 0
            compare_flag.append(a)
    mode_counts_ = []
    for i in range(len(mode_counts)):
        mode_counts_.append(mode_counts[i]/len(model))
    dataframe = pd.DataFrame(label_list)
    dataframe.to_csv(r"compare.csv")
    return count/len(mode_list), mode_counts, con_list, mode_list


def r_rate(model, label_list, mode_list):
    r_rate = []
    for i in range(len(model)):
        score = 0
        for j in range(len(label_list[i])):
            if label_list[i][j] == mode_list[j]:
                score += 1
        scores = score/len(label_list[i])
        r_rate.append(scores)
    print(len(r_rate))
    print(len(label_list))
    print(r_rate)
    return r_rate, np.argsort(r_rate)

def item_discrimination(r_rank, label_list, y_test, mode_list):
    top_rank = r_rank[:int(len(r_rank) * 0.27)]
    last_rank = r_rank[int(len(r_rank) * 0.73):]
    item_dis = []
    for i in range(len(y_test)):
        score1 = 0
        score2 = 0
        for j in range(len(top_rank)):
            if label_list[top_rank[j]][i] == mode_list[i]:
                score1 += 1
        scores1 = score1/len(top_rank)
        for k in range(len(last_rank)):
            if label_list[last_rank[k]][i] == mode_list[i]:
                score2 += 1
        scores2 = score2/len(last_rank)
        item_dis.append(scores1-scores2)
    print(item_dis)
    return item_dis, np.argsort(item_dis)

if __name__ == '__main__':
    import datetime
    start = datetime.datetime.now()
    # preprocess the data set
    from keras.datasets import fashion_mnist
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    from keras.models import load_model

    model = []
    model_name = '../models/model-'
    filename = []
    for i in range(1, 26):
        model_name_ = model_name + str(i) + '.h5'
        filename.append(model_name_)

    for i in range(len(filename)):
        model.append(load_model(filename[i]))
        print('model{} has been loaded.'.format(i+1))


    label_list = label_read(model, x_test, y_test)
    rate, mode_counts, con_list, mode_list = label_compare(model,label_list,y_test)
    r_rate, r_rank = r_rate(model, label_list, mode_list)
    r_rank = r_rank[::-1]
    item_dis, item_dis_rank = item_discrimination(r_rank, label_list, y_test, mode_list)
    print(item_dis_rank)
    print(item_dis[item_dis_rank[0]])
    item_dis_rank = item_dis_rank[::-1]
    rank_test = item_dis_rank[:int(len(item_dis_rank) * 0.25)]


    for k in range(50):
        print("the {} exp".format(k))
        acc_list = select_random(model,rank_test,x_test,y_test,5,30)
        for i in range(len(model)):
            np.savetxt('model{}/random{}.csv'.format(i+1,k), acc_list[i])


    end = datetime.datetime.now()
    print((end - start).seconds)

