
# RatMoSeq Extraction

This package is a refactored version of [MoSeq2-Extract](https://github.com/dattalab/moseq2-extract.git) from the Datta Lab at Harvard Medical School. In short, the library now uses Sam2 and keypoint tracking to segment the rodent of interest instead of traditional computer vision techniques. Please find details below for how to install the package and use it. 

# Installation 
## Conda environment creation
First, create and activate a new conda environment with `python>= 3.10`
```bash
conda create -n ratmoseq_extract python=3.10 -y
conda activate ratmoseq_extract
```
## Sam2 installation
Next, you will need to install Sam2 from facebook. This requires `torch>=2.5.1` and `torchvision>=0.20.1`. For installation instruction please follow the following [link](https://github.com/facebookresearch/sam2/tree/main?tab=readme-ov-file#installation)
## Check the Sam2 instllation worked!
If you are using a GPU, it is best to check that `torch` and Sam2 are actually installed
To check torch you can run the following code from the terminal with your conda environment activated:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If the above fails, please see the debugging docs TODO: link debugging
Next install your conda environment as a ipykernel so you can run the sam2 demo notebooks
```bash
pip install ipykernel
python -m ipykernel install --user --name ratmoseq_extract
```
To test that Sam2 worked, please follow the notebook [linked here](https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)
## FFMPEG installation
`ffmpeg` is a video processing library that will allow us to read/write videos, use the code snippet below to install it into your conda environment
```bash
conda install conda-forge::ffmpeg
```
## Installing the rest of the project
Now you're ready to install the rest of the project!
```bash
cd /path/to/ratmoseq_extract
pip install -e .
```
## Testing installation
Now that you have things installed locally, you can first check that it installed properly by running: `ratmoseq-extract --help`
You should see a help window that walks you through each of the commands you can use. 
To run extraction you can cd to a session directory and run the following command:
```
ratmoseq-extract extract depth.avi --sam2-checkpoint /path/to/sam2/checkpoints/sam2.1_hiera_tiny.pt --dlc-filename your_dlc_filename.csv --use-bground True --bground-type plane --num-frames 500
```
If you need to run DLC to get keypoints please check here to do so. There is also a notebook that will run DLC then extraction on all the data as well. 

# Usage
The pipeline works in the following steps:
1. Clip IR videos so they are between a given range of values [notebook here](TODO)
2. Perform keypoint estimation using DLC [notebook here](https://github.com/jackransomlovell/ratmoseq_extract/blob/main/notebooks/infer%20keypoints.ipynb)
3. Segment the rat using by telling Sam2 where it is in the frame with the keypoints provided [notebook here](https://github.com/jackransomlovell/ratmoseq_extract/blob/main/notebooks/extract%20all%20sessions.ipynb)

You will then have a subdirectory in each of your recordings called `proc`. That houses all the results from the extraction pipeline. Once those all exist you can run `ratmoseq-extract aggregate-results` to copy all the results to a new directory, and then proceed with the rest of the moseq pipeline starting with [pca](https://github.com/dattalab/moseq2-app/wiki/Command-Line-Interface-for-Extraction-and-Modeling#pca)
