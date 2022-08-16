# eilnn

# Overview of eilnn
*eil-nn* is an image processing toolkit to analyse tomography data for battery electrodes. It uses neural networks to segment the images and successively identify and classify particles. 

## Installation

Follow the steps given below to install the `eilnn` Python package. The package must be installed to run the included examples. It is recommended to create a virtual environment for the installation.

```bash
# Clone the repository
$ git clone https://github.com/sdaemi/eil-nn.git
# Got to the root directory
$ cd eil-nn
# Install the eilnn package from within the repository
$ pip install -e .
```

## Usage

The two example notebooks in the "examples" fodler indicate the two main functionalities of the package, namely automated COCO-style annotation generation and segmentation + classification. Please note for this iteration of package, Mask R-CNN (preferably GPU enabled) *must* be previously installed.

Currently, features such as the removal of partially truncated particles are tailored to data collected using a lab-based Zeiss Ultra 810 X-ray computed tomography instrument. If you need to analyse other types of data, please feel free to get in touch.

Both the pre-trained classifier model and Mask R-CNN weights are provided in the model/ folder for use with these tools or others.
