
import scripts.eval as eval
from scripts.network import *
import sys

model_name = sys.argv[0]
models = {
    "net1": Net1,
    "net2": Net2,
    "net3": Net3,
    "net4": Net4,
    "net5": Net5,
}
eval.predict_model("models/pascal3d-vp-cnn-"+model_name+".pth", models[model_name])
