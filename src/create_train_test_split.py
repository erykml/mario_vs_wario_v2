# script used for creating the train/test sets from the images

import glob
import os
from random import sample, seed
import shutil
from config import *

if __name__ == "__main__":
    
    seed(42)

    for class_name in CLASSES:
        files = glob.glob(f"{RAW_IMAGES_DIR}/{class_name}/*.jpg")

        sampled_indices = sample(range(len(files)), N_TRAIN + N_TEST)
        indices_dict = {
            "train": sampled_indices[:N_TRAIN],
            "test": sampled_indices[N_TRAIN:]
        }
        
        for dataset in ["train", "test"]:
            print(f"Creating {dataset} set for class {class_name}")
            
            for i in indices_dict[dataset]:
                
                f = os.path.basename(files[i])
                src = os.path.join(RAW_IMAGES_DIR, class_name, f)
                dst = os.path.join(PROCESSED_IMAGES_DIR, dataset, class_name, f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)