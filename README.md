# vqa-detection
Object detection-based Visual Question Answering

## Project status
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

## Dependencies

To execute this, you must have Python 3.6.*, [PyTorch](http://pytorch.org/), [OpenCV](http://opencv.org/), [Numpy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/) installed, to accomplish this, we recommend installing the [Anaconda](https://www.continuum.io/downloads) Python distribution and use conda to install the dependencies, as it follows:

```bash
conda install pytorch torchvision cuda80 -c soumith
conda install opencv -c conda-forge
conda install matplotlib numpy
conda install aria2 -c bioconda
pip install visual-genome
```

## Dataset download
You must download the [Visual Genome](http://visualgenome.org/api/v0/api_home.html) dataset, as well the train/val/test split used for our experiments. For this, we provide the ``download_dataset.sh`` bash script, it will take care of the downloads required.
