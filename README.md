# Bachelor's experiment

## Fetch repository

Fetch the current repository using
```bash
git clone https://github.com/d-pilipavicius/bach-ad-experiment.git
```

And after the repository is downloaded, run
```bash
cd bach-ad-experiment
```

## Set up environment

All dependencies used in this project are available inside the [environment.yml](environment.yml) file.

In order to automatically set up the python environment, use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) or any other Anaconda like distributions.

To build an environment for execution using conda, run

```bash
conda env create -f environment.yml
```

To use the built environment, run 
```bash
conda activate anomalib_dev
```

## Running the code

Before running experiments, change directory
```bash
cd src
```

In order to run the experiment, the MVTec LOCO AD dataset is required, which does not come bundled with this code. It is possible to download the dataset [here](https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz), or by running
```bash
wget https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz
```

Extract the dataset and place it under [src](/src/), or upload it where needed (configuration of dataset path can be changed in the [config.json](src/config.json) under the "dataset_path" field).

This code allows to train model from scratch, test already trained code by metrics images, and directly upload your own image for model running.

### Help

In order to see information about flags, run
```bash
python main.py -h
```

### Training

In order to train a model from scratch, run
```bash
python main.py TRAIN -n SETUP_NAME
```

To read more about SETUP_NAME, refer to [Setups](#setups) "name" explanation.

The trained model gets placed inside the "output" path, under the selected setup folder.

### Testing

In order to test a pre-trained model, run
```bash
python main.py TEST -n SETUP_NAME
```

The model statistics get saved under the "output" path inside of a txt file. Each tested image is outputed to the "results" path under the selected model. For example, if the PatchCore model for breakfast_box is used with the default results path, the images will be placed under `src/results/Patchcore/MVTecLOCO/breakfast_box/latest/images`,  

### Running model with specific image

To see how the model performs with a specific image, run
```bash
python main.py IMAGE -n SETUP_NAME -i IMAGE_PATH
```

The image is then placed under the current setup's directory in the "output" folder.

## Config.json configuration

[config.json](src/config.json) allows to change various configuration on how main.py operates. Here is a list of settable fields:

| Field | Mandatory | Description |
|---|---|---|
| cuda | Yes | When running models, changes whether they use the GPU or CPU. true for GPU, false for CPU. If no GPU is available, this will always be false. |
| worker_count | Yes | Allows to change how many workers are used when training models. For machines using Windows, it is recommended to leave this at 0. |
| dataset_path | No | MVTec LOCO AD dataset folder location. Default dataset path is `./datasets/MVTec_LOCO` if not filled (ANOMALIB default configuration). |
| model_dir | No | This folder is the default place ANOMALIB places everything related to model output. Default path if not filled is `./results`. |
| output_dir | Yes | This folder is custom made by the author to output models, metrics and images when running IMAGE mode. For example, if the program is using the "default" setup with PatchCore model and "breakfast_box" category, the program uses `{output_dir}/patchcore/breakfast_box/default` as its output path. |
| setup | Yes | Contains an array of setups. To read more about setups, refer to [Setups](#setups) |

### Setups
| Field | Mandatory | Description |
|---|---|---|
| name | Yes | Name of the setup. When running the model, the `-n SETUP_NAME` loads the setup containing the setup name. This is case insensitive. The default `SETUP_NAME` if no `-n` flag is provided is "default". |
| model | Yes | Name of the used model. Possible inputs: `cfa`, `efficientad`, `patchcore`. Case insensitive. |
| category | Yes | Anomaly detection models are trained only for a specific item. MVTec LOCO AD provides 5 categories: `breakfast_box`, `juice_bottle`, `pushpins`, `screw_bag`, `splicing_connectors`. Case sensitive. |
| batch_size | No | Sets the batch size for model training and validation. Default value is 32. |
| train_image_count | No | Used to train the model using a smaller subset of good examples from the training split. If left empty, full training split is used. |
| use_random_train_images | No | If true, select a random subset of the good training split. Only effective if train_image_count is set. |
| model_src | No | Used for TEST/IMAGE mode. Instead of using the "output_dir" model, this loads the model from the provided path. Provided path must match the "model" field of the used setup. |
| max_epochs | Sometimes | Changes how many epochs model trains for. Setups using the `patchcore` "model" type require "max_epochs" to be equal to 1. |
| trn_logical | No | Used to inject anomalous logical images into the training split. Provide a list of image names from the test split of the MVTec LOCO AD dataset. For example `"trn_logical": ["000.png"]`. Use only image name, not full path. If not provided, does not add any anomaly images to the training split. |
| trn_structural | No | Used to inject anomalous structural images into the training split. Provide a list of image names from the test split of the MVTec LOCO AD dataset. For example `"trn_structural": ["000.png"]`. Use only image name, not full path. If not provided, does not add any anomaly images to the training split. |

## Remove the environment after experimenting

When you are done with experimenting, run in order to remove the environment from your computer
```bash
conda env remove -n anomalib_dev
```