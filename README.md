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


### Project Team
Dr Shidong Wang, Newcastle University  ([Shidong.wang@newcastle.ac.uk](mailto:Shidong.wang@newcastle.ac.uk))  
Dr Elizabeth Lewis, Newcastle University  ([elizabeth.lewis2@newcastle.ac.uk](mailto:elizabeth.lewis2@newcastle.ac.uk))  

### RSE Contact
Robin Wardle  
RSE Team  
Newcastle University  
([robin.wardle@newcastle.ac.uk](mailto:robin.wardle@newcastle.ac.uk))  

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

### Installation

a. Install CUDA
https://developer.nvidia.com/cuda-downloads
removal
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" 
sudo apt-get --purge remove "*nvidia*"
sudo rm -rf /usr/local/cuda*

11.4 onwards
#### To uninstall cuda
sudo /usr/local/cuda-11.4/bin/cuda-uninstaller 
##### To uninstall nvidia
sudo /usr/bin/nvidia-uninstall

Version 10.2
https://developer.nvidia.com/cuda-10.2-download-archive

Won't install with GCC 9.3
sudo apt -y install gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

Then still won't install?


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


## Deployment

### Local

Deploying to a production style setup but on the local system. Examples of this would include `venv`, `anaconda`, `Docker` or `minikube`. 

### Production

This application is designed to be deployed to [DAFNI](https://dafni.ac.uk/). The HiPIMS repository (https://github.com/NCL-PYRAMID/PYRAMID-HiPIMS) can be used to create an Azure VM using the Terraform setup contained therein - see the HiPIMS repo README.md.

Having built a Docker images on an Azure VM, this will still need to be uploaded to DAFNI. Ensure that you have saved and zipped the image in a `.tar.gz` file, and then either use an FTP client such as [FileZilla](https://filezilla-project.org/) to transfer this image to your local computer for upload to DAFNI; or, alternatively, use the [DAFNI CLI](https://github.com/dafnifacility/cli) to uplad the model directly from the VM. The DAFNI API can also be used raw to upload the model, although the CLI embeds the relevant API calls within a Python wrapper and is arguably easier to use.

## Usage

Any links to production environment, video demos and screenshots.

## Roadmap

- [x] Data pre-processing
- [ ] Data and code are uploaded to [DAFNI platform](https://dafni.ac.uk/)   
- [ ] Test Docker 
- [ ] Online Visualisation  

## Contributing

### Main Branch
Protected and can only be pushed to via pull requests. Should be considered stable and a representation of production code.

### Dev Branch
Should be considered fragile, code should compile and run but features may be prone to errors.

### Feature Branches
A branch per feature being worked on.

https://nvie.com/posts/a-successful-git-branching-model/

## License

## Citiation

Please cite the associated papers for this work if you use this code:

```
@article{xxx2021paper,
  title={Title},
  author={Author},
  journal={arXiv},
  year={2021}
}
```


## Acknowledgements
This work was funded by a grant from the UK Research Councils, EPSRC grant ref. EP/L012345/1, “Example project title, please update”.
