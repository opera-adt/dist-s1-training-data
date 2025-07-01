# dist-s1-training-data

This repository provides curation of the DIST-S1 model.
See how this data is used to train a vision transformer to detect surface disturbances in [`dist-s1-model`](https://github.com/opera-adt/dist-s1-model).
This repository:

0. localizes the OPERA RTC-S1 data from the ASF DAAC
1. Finds spatial patches within a fixed Sentinel-1 burst that are used to construct co-registered time-series small enough to be analyzed by the vision transformer
2. Serializes the npz files for each data sample
3. Validates and visualizes the sample dataset.
4. Loads the data files into torch.

## Usage

Go through the notebooks.
