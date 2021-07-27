
import draw_bb
import cv2
import matplotlib.pyplot as plt
import numpy as np
import extraction
import os
import json


def as_cartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r = 1
    theta = rthetaphi[1]* np.pi/180 # to radian
    phi = rthetaphi[2]* np.pi/180
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return [x,y,z]


def create_cropped_dataset(width,height):
    DATASET_DIR_NAME = "imagenet_{}_{}_cropped_".format(width,height)

    for set in ["train", "val"]:
        dir = DATASET_DIR_NAME + set

        for cat in extraction.CATEGORIES:
            os.mkdir("../{}/{}_imagenet/".format(dir, cat))

        out_dict = {}
        for cat in extraction.CATEGORIES:
            out_dict[cat] = {}
            with open("../PASCAL3D+_release1.1/Image_sets/{}_imagenet_{}.txt".format(cat, set), "r") as f:
                for img_name in f.readlines():
                    img_name = img_name.replace("\n", "")
                    ann = extraction.get_image_annotations(
                        "../PASCAL3D+_release1.1/Annotations/{}_imagenet/{}.mat".format(cat, img_name), cat)[0]
                    az = ann["viewpoint"]["azimuth"]
                    el = ann["viewpoint"]["elevation"]
                    di = ann["viewpoint"]["distance"]
                    coords = as_cartesian([di, el, az])
                    out_dict[cat][img_name] = coords

        with open("../" + dir + "/annotation.json", "w") as f:
            f.write(json.dumps(out_dict))

        for cat in extraction.CATEGORIES:
            with open("../PASCAL3D+_release1.1/Image_sets/{}_imagenet_{}.txt".format(cat, set), "r") as f:
                for img_name in f.readlines():
                    img_name = img_name.replace("\n", "")
                    img = cv2.imread("../PASCAL3D+_release1.1/Images/{}_imagenet/{}.JPEG".format(cat, img_name))
                    sz = draw_bb.img_size(
                        "../PASCAL3D+_release1.1/Annotations/{}_imagenet/{}.mat".format(cat, img_name))
                    bb = draw_bb.get_bb("../PASCAL3D+_release1.1/Annotations/{}_imagenet/{}.mat".format(cat, img_name))
                    new_bb = draw_bb.adjust_aspect_ratio(sz, bb, width/height)
                    adj_img = draw_bb.resize_img(img, sz, new_bb)
                    adj_img = cv2.resize(adj_img, (width, height))
                    cv2.imwrite("../{}/{}_imagenet/{}.png".format(DATASET_DIR_NAME, cat, img_name), adj_img)


create_cropped_dataset(128,128)