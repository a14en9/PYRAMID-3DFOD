import os
import numpy as np
from utils import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Extract the project path
    pwd = os.path.dirname(os.path.realpath(__file__))
    # Directory containing .laz files
    input_laz_dir = os.path.join(pwd, "laz_format")
    if not os.path.exists(input_laz_dir):
        os.makedirs(input_laz_dir)
        print(f"{input_laz_dir} directory created.")

    # Directory to save .bbox info and covered_pcd files
    res_dir = os.path.join(pwd, "res")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        print(f"{res_dir} directory created.")

    # Directory to save .ply files
    output_ply_dir = os.path.join(pwd, "ply_format")
    if not os.path.exists(output_ply_dir):
        os.makedirs(output_ply_dir)
        print(f"{output_ply_dir} directory created.")

    # Check if the output directory already contains .ply files
    ply_files_exist = False
    for file_name in os.listdir(output_ply_dir):
        if file_name.endswith(".ply"):
            ply_files_exist = True
            break

    # Call the process_laz_files function only if the output directory doesn't contain .ply files
    if not ply_files_exist:
        process_laz_files(input_dir=input_laz_dir, output_dir=output_ply_dir)

    process_ply_files_in_directory(dir_ply_files=output_ply_dir, dir_res_files=res_dir)

    print("Finished-------------")









