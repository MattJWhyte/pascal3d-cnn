
import scripts.eval as eval
from scripts.network import *
import sys
import numpy as np

import matplotlib.cm as cm

model_details = sys.argv[1].lower()
model_name = model_details.split("-")[0]
width = int(sys.argv[2])
height = int(sys.argv[3])

eval.predict_model("models/pascal3d-vp-cnn-"+model_details+".pth", MODEL[model_name], model_details, (width,height))


'''
tr_vec, tr_cat, te_vec, te_cat = eval.get_model_thirty_deg_vector("models/pascal3d-vp-cnn-"+model_name+".pth",
                                                                  MODEL[model_name], (width,height))

np.save("results/{}/train-acc.npy".format(model_name), tr_vec)
np.save("results/{}/train-cat.npy".format(model_name), tr_cat)
np.save("results/{}/test-acc.npy".format(model_name), te_vec)
np.save("results/{}/test-cat.npy".format(model_name), te_cat)'''
