import matplotlib.pyplot as plt
import numpy as np
import json


CATEGORIES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]


def unweighted_accuracy(accuracy):
    return np.sum(accuracy)/accuracy.shape[0]


# Each function is meant to accept as input, a vector of 1/0 for accuracy scores by 30 degree angle,
# and an azimuth vector for each corresponding accuracy element
def bin_accuracy(accuracy, azimuth, size=30):
    steps = 360//size
    bin_ct = np.zeros(steps)
    bin_acc = np.zeros(steps)
    for i in range(0,azimuth.shape[0]):
        az = azimuth[i]
        bin_idx = int(((az+size/2.0)%360) // size)
        bin_ct[bin_idx] += 1.0
        bin_acc[bin_idx] += accuracy[i]
    ct = 0.0
    acc = 0.0
    for i in range(steps):
        if bin_ct[i] > 0:
            acc += bin_acc[i]/bin_ct[i]
            ct += 1
    return acc/ct


# Bin accuracy averaged over 15,30,45-degree bins
def avg_bin_accuracy(accuracy, azimuth):
    acc = sum([bin_accuracy(accuracy, azimuth, size=s) for s in [15, 30, 45]])
    return acc/3.0


# Take bin metric to the limit with bins of size 1-degree
def extreme_bin_accuracy(accuracy, azimuth):
    return bin_accuracy(accuracy, azimuth, size=1)


class DatasetInfo:
    def __init__(self, train=True):
        self.cat = np.load("../results/vgg_pose/train-cat.npy") if train else np.load("../results/vgg_pose/test-cat.npy")
        self.az = np.array(json.load(open("../az_data.txt","r"))["train" if train else "val"])
        self.dd = np.load("../weights/dd/train.npy", allow_pickle=True) if train else np.load("../weights/dd/val.npy", allow_pickle=True)


def get_accuracy(acc, info):
    uw_acc = unweighted_accuracy(acc)
    bin_acc = bin_accuracy(acc, info.az)
    ebin_acc = extreme_bin_accuracy(acc, info.az)
    dd_acc = np.dot(acc, info.dd)
    return np.array([uw_acc, bin_acc, ebin_acc, dd_acc])


def angle_in_bin(centre, width):
    def f(x):
        diff = width/2.0-centre
        if diff > 0:
            x = (x + diff)%360
        else:
            x = (x + 360+diff)%360
        return 1.0 if 0.0 <= x <= width else 0.0
    return f


def angle_between(a, b):
    def f(x):
        x = (x + (360-a)) % 360
        return 1.0 if 0 <= x <= (b + (360-a)) % 360 else 0.0
    return f

'''
weights/
        [weight_name]
                train/
                        full.npy
                        
'''

def continuous_bin_weighting(azimuth):
    w = np.zeros(azimuth.shape[0])
    for i in range(azimuth.shape[0]):
        ct = 0.0
        f = angle_in_bin(azimuth[i], 30)
        for j in range(azimuth.shape[0]):
            ct += f(azimuth[j])
        w[i] = 1.0/ct
    return w/sum(w)


def plot_simp_bins(info, st):
    for i, cat in enumerate(CATEGORIES):
        print(cat)
        b = info.cat[i]
        a = info.cat[i-1] if i > 0 else 0
        az_vec = info.az[a:b]
        cb_w_vec = continuous_bin_weighting(az_vec)
        score_ls = [[],[],[],[]]
        for j in range(360):
            bin_f = np.vectorize(angle_in_bin(j,30))
            acc_vec = bin_f(az_vec)
            uw_acc = unweighted_accuracy(acc_vec)
            bin_acc = bin_accuracy(acc_vec, az_vec)
            cont_bin_acc = np.dot(acc_vec, cb_w_vec)
            dd_acc = np.dot(acc_vec, np.load("../weights/dd/{}-{}.npy".format(cat,st), allow_pickle=True))
            score_ls[0].append(uw_acc)
            score_ls[1].append(bin_acc)
            score_ls[2].append(cont_bin_acc)
            score_ls[3].append(dd_acc)
        f = plt.figure()
        ax = f.add_subplot(1,1,1,projection='polar')
        deg = [np.deg2rad(i) for i in range(360)]
        ax.plot(deg, score_ls[0], label="unweighted")
        ax.plot(deg, score_ls[1], label="bins")
        ax.plot(deg, score_ls[2], label="cont. bins")
        ax.plot(deg, score_ls[3], label="density")
        ax.legend()
        plt.savefig("{}-eval.png".format(cat))
        plt.show()


az_data = json.load(open("../az_data.txt","r"))
train_az = np.array(az_data["train"])
test_az = np.array(az_data["val"])

az_data_by_cat = json.load(open("../az_data_by_cat.txt","r"))
train_az_by_cat = az_data_by_cat["train"]
test_az_by_cat = az_data_by_cat["val"]

train_acc = np.load("../results/vgg_pose/train-acc.npy")
train_cat = np.load("../results/vgg_pose/train-cat.npy")
test_acc = np.load("../results/vgg_pose/test-acc.npy")
test_cat = np.load("../results/vgg_pose/test-cat.npy")


for i,cat in enumerate(CATEGORIES):
    print(cat)
    cat_train_acc = train_acc[train_cat[i-1] if i > 0 else 0: train_cat[i]]
    cat_test_acc = test_acc[test_cat[i - 1] if i > 0 else 0: test_cat[i]]
    print("Unweighted")
    print("\tTrain: {}%".format(np.round(unweighted_accuracy(cat_train_acc), 3)))
    print("\tTest: {}%".format(np.round(unweighted_accuracy(cat_test_acc), 3)))
    print("Weighted 30-degree bin")
    print("\tTrain: {}%".format(np.round(bin_accuracy(cat_train_acc, np.array(train_az_by_cat[cat])), 3)))
    print("\tTest: {}%".format(np.round(bin_accuracy(cat_test_acc, np.array(test_az_by_cat[cat])), 3)))
    print("Weighted dd")
    print("\tTrain: {}%".format(np.round(np.dot(cat_train_acc, np.load("../weights/dd/{}-train.npy".format(cat),
                                                                     allow_pickle=True)), 3)))
    print("\tTest: {}%".format(np.round(np.dot(cat_test_acc, np.load("../weights/dd/{}-val.npy".format(cat),
                                                                     allow_pickle=True)), 3)))
    print("--------------")


print("WHOLE SET")
print("Unweighted")
print("\tTrain: {}%".format(np.round(unweighted_accuracy(train_acc), 3)))
print("\tTest: {}%".format(np.round(unweighted_accuracy(test_acc), 3)))
print("Weighted 30-degree bin")
print("\tTrain: {}%".format(np.round(bin_accuracy(train_acc, train_az), 3)))
print("\tTest: {}%".format(np.round(bin_accuracy(test_acc, test_az), 3)))
print("Continuous Weighted")
print("\tTrain: {}%".format(np.round(np.dot(train_acc, np.load("../weights/dd/train.npy",
                                                                     allow_pickle=True)), 3)))
print("\tTest: {}%".format(np.round(np.dot(test_acc, np.load("../weights/dd/val.npy",
                                                                     allow_pickle=True)), 3)))


train_info = DatasetInfo()
test_info = DatasetInfo(train=False)

#plot_simp_bins(train_info, "train")

train_acc_vals = np.load("../eval_datasets/biased_train_dataset/annotation.npy")
test_acc_vals = np.load("../eval_datasets/biased_val_dataset/annotation.npy")

train_acc_score = np.zeros(4)
test_acc_score = np.zeros(4)

for i in range(len(train_acc_vals)):
    if i % 20000 == 0:
        print("\n{}% train set samples complete ".format(int(np.round(float(i)/len(train_acc_vals)*100))))
        print("Current deviations : {}".format(train_acc_score/float(i)))
    elif i % 2000 == 0:
        print("XXX|", end='')
    train_acc = np.load("../eval_datasets/biased_train_dataset/sample-{}.npy".format(i))
    acc = get_accuracy(train_acc, train_info)
    train_acc_score += np.abs(acc - train_acc_vals[i]) # Get deviation
train_acc_score /= len(train_acc_vals)

for i in range(len(test_acc_vals)):
    if i % 20000 == 0:
        print("\n{}% test set samples complete ".format(int(np.round(float(i)/len(test_acc_vals)*100))))
        print("Current deviations : {}".format(test_acc_score/float(i)))
    elif i % 2000 == 0:
        print("XXX|", end='')
    test_acc = np.load("../eval_datasets/biased_val_dataset/sample-{}.npy".format(i))
    acc = get_accuracy(test_acc, test_info)
    test_acc_score += np.abs(acc - test_acc_vals[i]) # Get deviation
test_acc_score /= len(test_acc_vals)



