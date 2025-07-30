<!-- markdownlint-disable md029 -->

# Cat-Dog Image Classification

## Requirements

1. Create and activate a virtual environment that uses Python 3.12.
2. Install PyTorch 2.5.1 (see [this instruction](https://pytorch.org/get-started/previous-versions/)).
3. Install other dependencies: PIL 10.3.0, numpy 2.0.1, polars 1.21.0.
4. (For server deployment) onnx, onnxscript (install with pip) and [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server/tree/main). To run the client code at `ovms/client` you would need Node.js and NPM.

## Inference

### Directly using PyTorch model

See this [03_inference.ipynb](./notebooks/03_inference.ipynb) for more.

### Server

Do these steps:

1. Install [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server/tree/main) into the directory `ovms` of this repo (see the project's docs for instructions).

2. Start the server:

```sh
# Make sure the working directory is ovms
ovms --port 9000 --rest_port 8000 --model_name catdog --model_path models/catdog
```

3. After the server has been launched, run the client code

```sh
# Make sure the working directory is ovms/client
npm start ../../images/may-2.jpg

# Here "../../images/may-2.jpg" is simply the path to the image.
# Replace it with any images you want.

# This client code is a single Node.js script which only depends
# on sharp (see package.json), so it is portable.
```

## Training

How to train from scratch:

1. Download the dataset from [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=54765) (select English).
2. Extract and put the dataset folder under `data/`. If success, there should be two folders, `data/PetImages/Cat/` for cat images and `data/PetImages/Dog/` for dog images.
3. Run [01_training.ipynb](./notebooks/01_training.ipynb) to train the model on the training set and validates on the validation set.
4. Run [02_evaluation.ipynb](./notebooks/02_evaluation.ipynb) to evaluate the model on the testing set.
5. (Optional) Export your PyTorch model to ONNX format (see [04_export_onnx.ipynb](./notebooks/04_export_onnx.ipynb)).

Images need to be preprocessed before fed to the model (see the image below). See the transform functions in [01_training.ipynb](./notebooks/01_training.ipynb).

![Preprocess](./images/preprocess.png)

This is the plot of the losses and accuracy scores by epoch on the training and validation sets (see the image below).

![By epoch](./images/train.png)
