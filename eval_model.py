
import scripts.eval as eval
from scripts.network import *
import sys
import numpy as np

import matplotlib.cm as cm

model_name = sys.argv[1].lower()
width = int(sys.argv[2])
height = int(sys.argv[3])

train_mat, test_mat = eval.get_model_thirty_deg_vector("models/pascal3d-vp-cnn-"+model_name+".pth", MODEL[model_name], (width,height))

print(test_mat)

np.save("results/{}/train_acc_mat.npy".format(model_name), train_mat)
np.save("results/{}/test_acc_mat.npy".format(model_name), test_mat)

