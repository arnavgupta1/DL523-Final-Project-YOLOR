# DL523-Final-Project-YOLOR
Our task will be to implement the implicit knowledge capabilities from the paper “You Only Learn One Representation: Unified Network for Multiple Tasks” on the Faster R-CNN MobileNetV3-Large FPN network. We would be creating a baseline basic object detection and image classification network, then we will apply this paper which includes an implicit neural network and compare the performance between the two.

# Files

`dataloader.py` - defines the dataloaders to load in the miniCOCO dataset. This was edited from an existing file  
`implicit_layers.py` - defines the wrappers to add implicit knowledge after any given layer  
`models.py` - defines the model, which may be the baseline or with some or all implicit knowledge added  
`train_script.py` - trains and evaluates the model, saving the results in the results directory  
All files in the rcnn_utils file were modified from existing files

# Usage

1. Clone the repository
2. Download miniCOCO from `https://drive.google.com/file/d/1TLokPgnUNbd08uZ51m-7qvmN59u6Ud6y/view?usp=sharing` and unzip into project root folder 
3. run `python src/train_script.py -h` to see arguments
