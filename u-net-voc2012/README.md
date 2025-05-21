# U-Net for VOC Segmentation 2012

This project implements a U-Net model for semantic segmentation on the VOC 2012 dataset. The U-Net architecture is designed for biomedical image segmentation but can be applied to various segmentation tasks.

## Project Structure

- `configs/config.yaml`: Configuration file containing model training parameters such as learning rate, batch size, and number of epochs.
- `data/VOC2012`: Directory containing the original VOC 2012 dataset files.
- `data/processed`: Directory for storing processed data, including augmented or preprocessed images and labels.
- `models/unet.py`: Defines the structure of the U-Net model, including convolutional layers, pooling layers, and upsampling layers.
- `notebooks/exploratory.ipynb`: Jupyter Notebook for data exploration and visualization, including analysis of the dataset and preliminary evaluation of model performance.
- `src/dataset.py`: Defines the dataset class responsible for loading and preprocessing the VOC 2012 dataset, returning data for training and validation.
- `src/model.py`: Imports the U-Net model and defines functions related to training and evaluating the model.
- `src/train.py`: Training script containing the training loop, loss computation, and model saving logic.
- `src/utils.py`: Contains utility functions such as metric calculations and result visualizations.
- `requirements.txt`: Lists the required Python libraries and their versions for the project.

## Installation

To set up the project, clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare the VOC 2012 dataset by placing it in the `data/VOC2012` directory.
2. Modify the configuration parameters in `configs/config.yaml` as needed.
3. Run the training script:

```bash
python src/train.py
```

4. Explore the results and visualize them using the Jupyter Notebook in `notebooks/exploratory.ipynb`.

## Acknowledgments

This project is based on the U-Net architecture proposed by Olaf Ronneberger et al. in their paper "U-Net: Convolutional Networks for Biomedical Image Segmentation". The VOC 2012 dataset is a widely used benchmark for semantic segmentation tasks.