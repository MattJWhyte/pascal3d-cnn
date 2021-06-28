
import scripts.eval as eval
from scripts.network import *
import sys

import matplotlib.cm as cm

model_name = sys.argv[1].lower()
width = int(sys.argv[2])
height = int(sys.argv[3])

eval.predict_model("models/pascal3d-vp-cnn-"+model_name+".pth", MODEL[model_name], model_name, (width,height))
