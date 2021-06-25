
import scripts.eval as eval
from scripts.network import *
import sys

model_name = sys.argv[0]
eval.predict_model("models/pascal3d-vp-cnn-"+model_name+".pth", MODEL[model_name])
