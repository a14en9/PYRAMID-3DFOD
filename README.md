# PYRAMID-3DFOD

## About

This repo is about how to use machine learning algorithms like DBSCAN to cluster 3D point cloud data (collected by using CityMapper)
and draw oriented 3D bounding boxes on each cluster (basically different vehicles in this case ). This can be achieved by using the following steps:

- Load input LAZ files and extract points that are labelled as vehicles, and save the extracted points into new PLY files.
- All PLY files needed to be pre-prcossed by using methods avaiable in `utils.py`.
- Use DBSCAN algorithm to cluster the processed data while drawing oriented 3D bounding boxes.
- Output the boxes' information (i.e., `Oriented Bounding Boxes`, `Centre Coordinates of Boxes`, `Max XYZ Values of Boxes`, `Min XYZ Values of Boxes`, `Dimensions of Boxes` and `Points Covered by Boxes`) and save the points covered by all boxes into array. 

An examplar visualisation of the ouput result is shown below:

<img src="3dfod.gif" width="60%">


## Built With

This section is intended to list the frameworks and tools you're using to develop this software. Please link to the home page or documentatation in each case.

[Open3D](http://www.open3d.org/docs/release/)  

## Getting Started

### Prerequisites

The dependent libs can be found in the [requirements.txt](requirements.txt). Specifically, it needs:
- Linux
- Python 3.9 
- open3d
- laspy
- matplotlib (optional)
- scikit-learn (optional)

a. Create a conda virtual environment and activate it. 

```shell
conda create -n 3dfod python=3.9 -y
source activate 3dfod
```

b. Clone this repository (Skip this step if the repo exists locally).

```shell
git clone https://github.com/cvhabitat/PYRAMID-3DFOD.git
cd PYRAMID-3DFOD
```

c. Install all requirements.

```shell
pip install -r requirements.txt
```

### Running

Run the command below and the results will be generated in the txt format and saved to a directory named `res`.
```shell
python 3DFOD.py
```

### Running Locally
a. Organise the data and scripts as the following structure:

```bash
├─ laz_format                           # Data folder where all original laz files saved
    ├─ NZ2463.laz                       
    ├─ ...
├─ ply_format                           # Data folder where all extracted cars data saved
    ├─ NZ2463.ply                       
    ├─ ...
├─ res                                  # Data folder to save the generated results
    ├─ NZ2463.txt                       
    ├─ ...
├─ 3DFOD.py                             # Main function 
├─ ply.py                               # Functions to read/write .ply files
├─ Dockerfile                           # Docker script
├─ pts.py                               # Function to read/write points
├─ requirements.txt                     # List all envs that need to be downloaded and installised
├─ utils.py                             # Functions to process point cloud data
```
