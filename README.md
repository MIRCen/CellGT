# CellGT

Project for automatic classification of neurons per anatomical region in entire brain slices marked with the biomarker NeuN. 

# Installation

This project uses [Pixi](https://prefix.dev/docs/pixi/) for managing a reproducible environment. 
First, install `pixi`:
```bash
curl -sSL https://prefix.dev/install.sh | bash

Install the environment: pixi install
Activate the environment: pixi shell

# Dataset of 




# Run neuron classification
python test_classification_model.py

# Run region segmentation
python test_region_reconstruction.py

# CellGT training 
python training_model.py
