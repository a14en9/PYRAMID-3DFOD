# PYRAMID-3DFOD

## About

This repo is about how to use machine learning algorithms like DBSCAN to cluster 3D point cloud data (collected by using CityMapper)
and draw oriented 3D bounding boxes on each cluster (basically different vehicles in this case ). 

An examplar visualisation of the ouput result is shown below:

<img src="3dfod.gif" width="60%">

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
b. Run the command below and the results will be generated in the txt format and saved to a directory named `res`.
```shell
python 3DFOD.py
```


## Roadmap

- [x] Data pre-processing
- [ ] Data and code are uploaded to [DAFNI platform](https://dafni.ac.uk/)   
- [ ] Test Docker 
- [ ] Online Visualisation  
