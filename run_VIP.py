# Some basic setup:
# Setup detectron2 logger
"""
['/data4/ersp2022/models/Detic', '/data4/ersp2022/miniconda3/envs/videocot/lib/python39.zip', '/data4/ersp2022/miniconda3/envs/videocot/lib/python3.9', '/data4/ersp2022/miniconda3/envs/videocot/lib/python3.9/lib-dynload', '/data4/ersp2022/miniconda3/envs/videocot/lib/python3.9/site-packages', '/data4/ersp2022/models/detectron2']
['/data4/ersp2022/vaishnavi/musharna', '/data4/ersp2022/miniconda3/envs/videocot/lib/python39.zip', '/data4/ersp2022/miniconda3/envs/videocot/lib/python3.9', '/data4/ersp2022/miniconda3/envs/videocot/lib/python3.9/lib-dynload', '/data4/ersp2022/miniconda3/envs/videocot/lib/python3.9/site-packages', '/data4/ersp2022/models/detectron2']
['third_party/CenterNet2/', '/data4/ersp2022/models/Detic', '/data4/ersp2022/miniconda3/envs/videocot/lib/python39.zip', '/data4/ersp2022/miniconda3/envs/videocot/lib/python3.9', '/data4/ersp2022/miniconda3/envs/videocot/lib/python3.9/lib-dynload', '/data4/ersp2022/miniconda3/envs/videocot/lib/python3.9/site-packages', '/data4/ersp2022/models/detectron2']
"""
import sys

# sys.path.insert(0, "/data4/ersp2022/models")
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import scipy.stats as stats
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(
    0, "/data4/ersp2022/models/Detic/third_party/CenterNet2"
)  # Add path to CenterNet2
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test

# Build the detector and download our pretrained weights
cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file(
    "/data4/ersp2022/models/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
)  # Add path to CLIP config
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = (
    True  # For better visualization purpose. Set to False for all classes.
)
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.

predictor = DefaultPredictor(cfg)
# sys.path.append("/data4/ersp2022/models/Detic")
# Setup the model's vocabulary using build-in datasets

BUILDIN_CLASSIFIER = {
    "lvis": "datasets/metadata/lvis_v1_clip_a+cname.npy",
    "objects365": "datasets/metadata/o365_clip_a+cnamefix.npy",
    "openimages": "datasets/metadata/oid_clip_a+cname.npy",
    "coco": "datasets/metadata/coco_clip_a+cname.npy",
}

BUILDIN_METADATA_PATH = {
    "lvis": "lvis_v1_val",
    "objects365": "objects365_v2_val",
    "openimages": "oid_val_expanded",
    "coco": "coco_2017_val",
}

vocabulary = "lvis"  # change to 'lvis', 'objects365', 'openimages', or 'coco'
metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)
# wget "https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg"

# Run model and show results
# im = cv2.imread("/data4/ersp2022/images/test.jpg")
# im2 = cv2.imread(
#     "'/data4/ersp2022/danny/musharna/keyframes/1324866/HugeThunderstormFormsOverKansas-1324866-watermarked_9.jpeg'"
# )
# images = os.listdir("/data4/ersp2022/keyframes5")
# print(images)
"""
for image in images:
    im = cv2.imread(f"/data4/ersp2022/keyframes5/{image}")
    outputs = predictor(im)
    print(image)
    v = Visualizer(im[:, :, ::-1], metadata)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow(out.get_image()[:, :, ::-1])

    #print(outputs["instances"].pred_classes) # class index
    #print([metadata.thing_classes[x] for x in outputs["instances"].pred_classes.cpu().tolist()]) # class names
    objects = [metadata.thing_classes[x] for x in outputs["instances"].pred_classes.cpu().tolist()]
    boxes = outputs["instances"].pred_boxes
    scores = outputs["instances"].scores
    #print("scores", scores)
    scores = stats.zscore(scores.cpu().numpy())
    with open('/data4/ersp2022/data/scene_descriptions_pilot/video5.txt', 'a') as f:
        f.write(f"Image name: {image}\n")
        f.write("Object Detection Model Ouput:\n")
        for obj, score in zip(objects, scores):
            for row in boxes:
                x1 = row[0]
                y1 = row[1]
                x2 = row[2]
                y2 = row[3]
            f.write(f"{obj} x1: {x1:.2f} x2: {x2:.2f} y1: {y1:.2f} y2: {y2:.2f} score: {score:.3f}\n")
        f.write("\n")
    #print(outputs["instances"].pred_boxes)
"""


def runDetic_VIP(images_folder: str) -> dict:
    """
    Runs detic on all images in the images_folder and returns a dictionary with keys as the image names and values as the object values and positions
    """
    output = {}
    images = os.listdir(images_folder)
    for image in images:
        im = cv2.imread(f"{images_folder}/{image}")
        d_output = predictor(im)
        objects = [
            metadata.thing_classes[x]
            for x in d_output["instances"].pred_classes.cpu().tolist()
        ]
        boxes = d_output["instances"].pred_boxes
        scores = d_output["instances"].scores
        scores = stats.zscore(scores.cpu().numpy())
        output[image] = ""
        for obj, score in zip(objects, scores):
            for row in boxes:
                x1 = row[0]
                y1 = row[1]
                x2 = row[2]
                y2 = row[3]
            output[
                image
            ] += f"{obj} x1: {x1:.2f} x2: {x2:.2f} y1: {y1:.2f} y2: {y2:.2f} score: {score:.3f}\n"
    return output


def runDeticKeyFrame_VIP(images_folder: str) -> dict:
    """
    Runs detic on all images in the images_folder and returns a dictionary with keys as the image names and values as the object values in one sentence.
    """
    output = {}
    images = os.listdir(images_folder)
    for image in images:
        im = cv2.imread(f"{images_folder}/{image}")
        d_output = predictor(im)
        objects = [
            metadata.thing_classes[x]
            for x in d_output["instances"].pred_classes.cpu().tolist()
        ]
        objects = list(set(objects))
        outputList = ["", False, 0]
        if "person" in objects:
            outputList[1] = True
        if len(objects) > 25:
            objects = objects[0:24]
        outputList[2] = len(objects)
        objectListString = ""
        for i in range(len(objects)):
            objectListString += f"{objects[i]}"
            if i != (len(objects) - 1):
                objectListString += ", "
        outputList[0] = objectListString
        output[image] = outputList
    return output


if __name__ == "__main__":
    print(
        runDetic_VIP("/data4/ersp2022/danny/exampleimages/objects")
    )  # Put path to images folder
