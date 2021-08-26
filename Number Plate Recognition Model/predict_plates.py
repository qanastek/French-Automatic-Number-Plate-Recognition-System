import argparse
import os.path

from transformers import DetrFeatureExtractor, DetrForObjectDetection
from hugsvision.inference.ObjectDetectionInference import ObjectDetectionInference

parser = argparse.ArgumentParser(description='Image classifier')
parser.add_argument('--path', type=str, default="./out/MYDETRMODELLICENCEPLATES/10_2021-08-26-01-39-07/", help='The model path')
parser.add_argument('--img', type=str, default="/users/ylabrak/datasets/FrenchLicencePlateDataset/coco/test/198 GV 73.jpg", help='The input image')
parser.add_argument('--threshold', type=float, default=0.25)
args = parser.parse_args() 

print("Process the image: " + args.img)

try:
        
    inference = ObjectDetectionInference(
        DetrFeatureExtractor.from_pretrained(args.path),
        DetrForObjectDetection.from_pretrained(args.path, from_tf=False)
    )

    inference.predict(
        args.img,
        threshold = args.threshold
    )

except Exception as e:
    if "containing a preprocessor_config.json file" in str(e) and os.path.isfile(args.path + "config.json") == True:
        print("\033[91m\033[4mError:\033[0m")
        print("\033[91mRename the config.json file into \033[4mpreprocessor_config.json\033[0m")
    else:
        print(str(e))