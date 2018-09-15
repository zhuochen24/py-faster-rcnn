# py-faster-rcnn with virtual pooling. 

## Installation

Follow py-faster-rcnn official installation guide

## Usage

Sensitivity analysis:

Run 'python ./models/pascal\_voc/VGG16/faster\_rcnn\_end2end/modify\_net.py' to generate model files with interpolation layers embedded.

Save the output lines in vgg16\_lininterp\_perlayer.sh and run './experiments/scripts/interp\_faster\_rcnn\_end2end\_perlayer.sh' to do sensitivity analysis in batch mode.
