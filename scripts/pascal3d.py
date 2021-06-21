import scipy.io as sio
import os

IMAGESET_DIR = "../PASCAL3D+_release1.1/Image_sets/"
ANNOTATION_DIR = "../PASCAL3D+_release1.1/Annotations/"
CATEGORIES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]


def get_samples(dataset, category):
    return [f.path for f in os.scandir(ANNOTATION_DIR + category + "_" + dataset)]


def get_imageset(dataset, category, subset):
    with open(IMAGESET_DIR + "{}_{}_{}.txt".format(category, dataset, subset), "r") as f:
        return [ANNOTATION_DIR + "{}_{}/".format(category, dataset) + s.replace("\n", "") for s in f.readlines()]


# Get annotation information for all objects of specified category in image
def get_image_annotations(img, category):
    mat = sio.loadmat(img)
    objs = mat["record"][0, 0]["objects"]
    annotation_list = []
    for i in range(objs.shape[1]): # Loop over objects in image
        o = objs[0, i]
        if len(o.dtype) > 0:
            if o["class"][0] != category:
                continue
            annotation_dict = {
                "truncated": o["truncated"][0, 0],
                "occluded": o["occluded"][0, 0],
                "difficult": o["difficult"][0, 0],
                "class": o["class"][0]
            }
            vp = o["viewpoint"]
            if vp.size > 0:
                vp_annotation = {}
                for annotation in vp.dtype.fields.keys():
                    vp_annotation[annotation] = vp[annotation][0, 0][0, 0]
                annotation_dict["viewpoint"] = vp_annotation
                annotation_list.append(annotation_dict)
            else:
                print("WARNING : no viewpoint set for annotation")
                continue
    return annotation_list


def extract_annotations_by_condition(img_list, category, field, cond, include_coarse=False):
    true_cond_list = []
    false_cond_list = []
    for img in img_list:
        annotation_list = get_image_annotations(img, category)
        for annotation_dict in annotation_list:
            if annotation_dict["viewpoint"]["num_anchor"] > 0:
                if cond(annotation_dict):
                    true_cond_list.append(annotation_dict["viewpoint"][field])
                else:
                    false_cond_list.append(annotation_dict["viewpoint"][field])
            elif include_coarse:
                if cond(annotation_dict):
                    true_cond_list.append(annotation_dict["viewpoint"][field + "_coarse"])
                else:
                    false_cond_list.append(annotation_dict["viewpoint"][field + "_coarse"])
    return true_cond_list, false_cond_list
