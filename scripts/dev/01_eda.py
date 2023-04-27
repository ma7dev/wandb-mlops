from pathlib import Path
import numpy as np
from PIL import Image
from fastai.data.external import untar_data
from fastai.data.transforms import get_image_files
from fastprogress import progress_bar

import wandb

import src
import src.params as params

PROJECT_PATH = Path(src.__file__).parent.parent
OUTPUT_PATH = f"{PROJECT_PATH}/output/01_eda"

URL = 'https://storage.googleapis.com/wandb_course/bdd_simple_1k.zip'

def label_func(fname):
    return f"{fname.parent.parent}/labels/{fname.stem}_mask.png"

def get_classes_per_image(mask_data, class_labels):
    unique = list(np.unique(mask_data))
    result_dict = {}
    for _class in class_labels.keys():
        result_dict[class_labels[_class]] = int(_class in unique)
    return result_dict

def _create_table(image_files, class_labels):
    "Create a table with the dataset"
    
    # get the labels
    labels = [str(class_labels[_lab]) for _lab in list(class_labels)]
    
    # create a table with the dataset
    table = wandb.Table(columns=["File_Name", "Images", "Split"] + labels)
    
    # add the data to the table
    for i, image_file in progress_bar(enumerate(image_files), total=len(image_files)):
        # get the image and the mask
        image = Image.open(image_file)
        mask_data = np.array(Image.open(label_func(image_file)))
        
        # get the classes in the image
        class_in_image = get_classes_per_image(mask_data, class_labels)
        
        # add the data to the table
        table.add_data(
            # file name
            str(image_file.name),
            
            # image
            wandb.Image(
                    image,
                    masks={
                        "predictions": {
                            "mask_data": mask_data,
                            "class_labels": class_labels,
                        }
                    }
            ),
            "None", # we don't have a dataset split yet
            *[class_in_image[_lab] for _lab in labels]
            # TODO: why is this needed?
        )
    
    return table

if __name__ == "__main__":
    path = Path(untar_data(URL, force_download=True))
    run = wandb.init(
        project=params.WANDB_PROJECT, 
        entity=params.ENTITY, # team name
        job_type="upload",
        dir=OUTPUT_PATH
    )
    raw_data_at = wandb.Artifact(
        params.RAW_DATA_AT, 
        type="raw_data"
    )

    raw_data_at.add_file(f'{path}/LICENSE.txt', name='LICENSE.txt')

    raw_data_at.add_dir(f'{path}/images', name='images')
    raw_data_at.add_dir(f'{path}/labels', name='labels')

    image_files = get_image_files(f'{path}/images', recurse=False)

    table = _create_table(image_files, params.BDD_CLASSES)

    raw_data_at.add(table, "eda_table")

    # upload the information to wandb
    run.log_artifact(raw_data_at)
    run.finish()