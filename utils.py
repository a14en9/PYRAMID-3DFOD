import os
import laspy
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import open3d as o3d
from ply import *
from pts import *
import plyfile
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score


def process_ply_file(file_path, output_dir):
    # load ply file and process it
    pc_coordinates, pc_label = load_ply(file_path)
    input_pcd = create_pcd(pc_coordinates)
    input_pcd = non_finite_points_removal(duplicated_points_removal(input_pcd))
    ortho_pcd = make_orthogonal_pcd(input_pcd)

    voxel_size = 0.05
    point_cloud_downsampled = ortho_pcd.voxel_down_sample(voxel_size)
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
    point_cloud_downsampled.estimate_normals(search_param=search_param)

    plane_model, inliers = point_cloud_downsampled.segment_plane(distance_threshold=0.05, ransac_n=3,
                                                                 num_iterations=100)
    outlier_cloud = point_cloud_downsampled.select_by_index(inliers, invert=True)

    depsilon = density_based_eps(outlier_cloud)
    boxes, box_centres, box_dims, covered_points = draw_obbx_from_clusters(outlier_cloud, depsilon, min_points_box=5,
                                                                           visual_clusters=True, visual_per_box=False, visual_all_boxes=True, visual_covered_points=True)

    # save results to file
    filename = os.path.splitext(os.path.basename(file_path))[0]
    # output_dir = "res"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, filename + ".txt")
    with open(output_file, "w") as f:
        for box in boxes:
            f.write("box:\n")
            f.write(str(box) + "\n")
        f.write("box_centres:\n")
        f.write(str(box_centres) + "\n")
        f.write("box_dims:\n")
        f.write(str(box_dims) + "\n")
        f.write("covered_points:\n")
        for points in covered_points:
            f.write(str(points) + "\n")

    print(f"Processed file {file_path} and saved results to {output_file}")


def process_ply_files_in_directory(dir_ply_files, dir_res_files):
    for root, dirs, files in os.walk(dir_ply_files):
        for file in files:
            if file.endswith(".ply"):
                file_path = os.path.join(root, file)
                process_ply_file(file_path, dir_res_files)

def make_orthogonal_pcd(pcd):
    # Get the bounding box of the point cloud
    bbox = pcd.get_axis_aligned_bounding_box()

    # Compute the center of the bounding box
    center = bbox.get_center()

    # Compute the rotation matrix to align the bounding box axes with the x, y, and z axes
    R = bbox.get_rotation_matrix_from_xyz((0, 0, 0))

    # Compute the translation vector to move the center of the bounding box to the origin
    T = -center

    # Apply the rotation and translation to the point cloud
    pcd.rotate(R)
    pcd.translate(T)

    # Return the transformed point cloud
    return pcd


def process_laz_files(input_dir, output_dir):
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Recursively traverse the input directory and its subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check if the file is a laz file
            if file.endswith(".laz"):
                # Construct the input and output file paths
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_dir, os.path.splitext(file)[0] + ".ply")

                # Process the laz file
                with laspy.open(input_file, mode="r") as f:
                    # Load the points
                    las = f.read()
                    # Extract the classification field
                    classification = las.classification
                    # Create a boolean mask to select the points with the desired label
                    mask = classification == 20
                    # Extract the selected points
                    selected_points = las.points[mask]
                    # selected_xyz = las.xyz[mask]  ### Use this if you only want to extract the points' coordinates
                    # Get the scaling factors
                    x_scale, y_scale, z_scale = las.points.scales
                    # Extract the x, y, and z coordinates of the selected points
                    x = selected_points['X'] * x_scale
                    y = selected_points['Y'] * y_scale
                    z = selected_points['Z'] * z_scale
                    # Convert 3D coordinates and the features of the selected point cloud to a numpy array
                    # Below two lines can be selectively used

                    # vehicle_coords = np.asarray(np.vstack((x, y, z)).transpose())
                    vehicle_feats_intensity, vehicle_feats_return_number, vehicle_feats_number_of_returns = np.array(
                        selected_points['intensity'], dtype=np.uint8), \
                        np.array(selected_points['return_number'], dtype=np.uint8), np.array(
                        selected_points['number_of_returns'], dtype=np.uint8)

                    # Generate pseudo labels for the selected points just in case
                    vehicle_labels = np.ones(len(selected_points), dtype='uint8')
                    # Create a dict for saving the selected points into .ply file
                    data_ply = {
                        "x": np.asarray(x),
                        "y": np.asarray(y),
                        "z": np.asarray(z),
                        "intensity": vehicle_feats_intensity,
                        "return_number": vehicle_feats_return_number,
                        "number_of_returns": vehicle_feats_number_of_returns,
                        "labels": vehicle_labels,
                    }

                    # Save preprocessed point cloud
                    path_ply = output_file
                    dict2ply(data_ply, path_ply)

                    print(f"PLY point cloud successfully saved to {path_ply}")


def load_ply(file):
    ########## Load in the point cloud data
    plydata = plyfile.PlyData.read(file)
    ######### Adjust data that can be processed and visualised by using open3d
    # Extract x, y, and z coordinates of the points
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    pcd = np.column_stack((x, y, z))
    # vehicle_coordinates = np.stack((pcd.elements[0].data['x'], pcd.elements[0].data['y'], pcd.elements[0].data['z']), axis=1)
    ########## Load the vehicle label
    vehicle_label = plydata.elements[0].data['labels']
    return pcd, vehicle_label


def create_pcd(coordinates):
    ########## Initialise the visualisation
    # print("Visualising the point cloud data")
    pcd = o3d.geometry.PointCloud()
    ########## Convert coordinates to the Vector format
    pcd.points = o3d.utility.Vector3dVector(coordinates.reshape(-1, 3))
    return pcd


def visualise_pcd(pcd):
    ########## Set colour for the points if needed
    ######## pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    ########## Visualising the point cloud
    o3d.visualization.draw_geometries([pcd])


def crop_pcd(pcd):
    ########## Crop the input point cloud into chunks
    ########## Define the ROI using the min and max coodrinates
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=pcd.get_min_bound(), max_bound=pcd.get_center())
    ######## Crop the point cloud
    cropped_pcd = pcd.crop(bbox)
    return cropped_pcd


def voxel_downsampling(pcd, voxel_size):
    ########## Voxel-based downsampling
    # print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # o3d.visualization.draw_geometries([pcd_down])
    return downpcd


def uniform_downsampling(pcd, every_k_points):
    ########## Uniformlly downsample
    # print("Downsample the point cloud uniformlly")
    downpcd = pcd.uniform_down_sample(every_k_points=every_k_points)
    # o3d.visualization.draw_geometries([pcd_down])
    return downpcd


def farthest_downsampling(pcd, factor):
    ########## Farthest point downsample
    # print("Downsample the point cloud with a set of points has farthest distance.")
    downpcd = pcd.farthest_point_down_sample(int(factor * len(np.asarray(pcd.points))))
    # o3d.visualization.draw_geometries([pcd_down])
    return downpcd


def random_downsampling(pcd, sampling_ratio):
    ########## Randomly downsample
    # print("Downsample the point cloud randomly")
    downpcd = pcd.random_down_sample(sampling_ratio)
    # o3d.visualization.draw_geometries([pcd_down])
    return downpcd


def radius_outliers_removal(pcd, nb_points, radius, invert=False):
    ########## Function to remove points that have less than nb_points in a given sphere of a given radius
    ########## Radius outlier removal:
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    inlier_rad_pcd = pcd.select_by_index(ind, invert=invert)
    # outlier_rad_pcd.paint_uniform_color([0., 0., 1.])
    # o3d.visualization.draw_geometries([outlier_rad_pcd])
    return inlier_rad_pcd


def statistic_outliers_removal(pcd, nb_neighbors, std_ratio, invert=False):
    ########## Function to remove points that are further away from their neighbors in average.
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_rad_pcd = pcd.select_by_index(ind, invert=invert)
    # outlier_rad_pcd.paint_uniform_color([0., 0., 1.])
    # o3d.visualization.draw_geometries([outlier_rad_pcd])
    return inlier_rad_pcd


def duplicated_points_removal(pcd):
    ########## Removes duplicated points, i.e., points that have identical coordinates.
    cleaned_pcd = pcd.remove_duplicated_points()
    return cleaned_pcd


def non_finite_points_removal(pcd, remove_nan=True, remove_infinite=True):
    ######### Removes all points from the point cloud that have a nan entry, or infinite entries.
    cleaned_pcd = pcd.remove_non_finite_points(remove_nan=remove_nan, remove_infinite=remove_infinite)
    return cleaned_pcd


def normal_estimation(pcd, radius, max_nn):
    ########## Point normal estimation. This operator could be applied to both undownsamplped and downsampled points
    print("Recompute the normal of the downsampled point cloud")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    return pcd


def resample_point_cloud(pcd):
    # Determine target voxel size based on point cloud density
    num_points = len(pcd.points)
    point_cloud_volume = pcd.get_max_bound() - pcd.get_min_bound()
    point_density = num_points / np.prod(point_cloud_volume)

    # Set a minimum voxel size to prevent oversampling
    MIN_VOXEL_SIZE = 0.01

    # Compute the target voxel size as the cube root of the inverse point density
    target_voxel_size = max(MIN_VOXEL_SIZE, 1 / (point_density ** (1 / 3)))

    # Resample point cloud at a uniform density using Open3D's uniform_down_sample() function
    downsampled_pcd = pcd.uniform_down_sample(every_k_points=int(target_voxel_size))

    return downsampled_pcd


def normalize_point_cloud(pcd):
    # Compute centroid
    centroid = np.mean(pcd.points, axis=0)

    # Translate to origin
    pcd.translate(-centroid)

    # Compute max distance from origin
    max_distance = np.max(np.linalg.norm(pcd.points, axis=1))

    # Scale to unit sphere
    pcd.scale(1.0 / max_distance, center=[0, 0, 0])

    return pcd


def denormalize_point_cloud(pcd, centroid, max_distance):
    # Scale back to original size
    pcd.scale(max_distance, center=[0, 0, 0])

    # Translate back to original position
    pcd.translate(centroid)

    return pcd


def find_elbow_point(x, y):
    # Define a window_size which is normally in 5%-20% of the given data
    window_size = int(0.15 * len(x))
    # Apply Savitzky-Golay filter to smooth the curve
    # smoothed = savgol_filter(y, 51, 3)
    smoothed_distance = np.convolve(x, np.ones(window_size) / window_size, mode='same')
    # Calculate the second derivative
    d2 = np.gradient(np.gradient(smoothed_distance))

    # Find the index of the maximum second derivative
    elbow_idx = np.argmax(d2)

    return x[elbow_idx], y[elbow_idx]


def compute_knee_value(pcd, k_value, plot_kdist=True):
    # Compute the k-distance plot
    nbrs = NearestNeighbors(n_neighbors=k_value).fit(pcd.points)
    distances, indices = nbrs.kneighbors(pcd.points)
    k_distances = np.mean(distances[:, 1:], axis=1)
    kth_distances = np.sort(k_distances, axis=0)[::-1]
    if plot_kdist:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(kth_distances)), kth_distances)
        ax.set_xlabel('Points sorted by distance')
        ax.set_ylabel(f'{k_value}-NN distance')
        ax.grid(True)
        plt.show()

    # Find the elbow point of the k-distance plot
    elbow_x, elbow_y = find_elbow_point(np.arange(len(kth_distances)), kth_distances)
    eps = elbow_y
    """
    # Determine knee value of plot to get optimal eps value
    deltas = np.diff(k_distances)
    knee_idx = np.argmax(deltas) + 1  # Add 1 because of diff
    eps = k_distances[knee_idx]
    # print("Optimal eps value: {}".format(eps))  # Print estimated eps values
    """
    return eps


def compute_elbow_values(point_cloud, k_values, plot_kdist=True):
    # Initialize dictionary to store knee values
    elbow_values = {}
    pcd_array = np.asarray(point_cloud.points)
    for k in k_values:
        # Build KDTree using point cloud data
        tree = KDTree(pcd_array)

        # Calculate k-distance for each point
        distances, indices = tree.query(pcd_array, k=k + 1)
        k_distance = distances[:, -1]

        # Sort k-distances in ascending order
        k_distance_sorted = np.sort(k_distance)
        if plot_kdist:
            # Plot k-distances
            plt.plot(k_distance_sorted)
            plt.title("K-distance plot")
            plt.xlabel("Points")
            plt.ylabel("K-distance")
            plt.show()

        # Find the knee point in the graph
        knee = np.diff(k_distance_sorted, 2)
        knee_point = knee.argmax() + 2

        # Get the epsilon value
        epsilon = k_distance_sorted[knee_point]
        # Store knee value in dictionary
        elbow_values[k] = epsilon

    return elbow_values


def compute_knee_values(pcd, k_values, plot_kdist=True):
    # Initialize dictionary to store knee values
    knee_values = {}

    for k in k_values:
        # Compute the k-distance plot
        nbrs = NearestNeighbors(n_neighbors=k).fit(pcd.points)
        distances, indices = nbrs.kneighbors(pcd.points)
        k_distances = np.mean(distances[:, 1:], axis=1)
        kth_distances = np.sort(k_distances, axis=0)[::-1]
        if plot_kdist:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(len(distances)), distances)
            ax.set_xlabel('Points sorted by distance')
            ax.set_ylabel(f'{k}-NN distance')
            ax.grid(True)
            plt.show()

        # Find the elbow point of the k-distance plot
        elbow_x, elbow_y = find_elbow_point(np.arange(len(kth_distances)), kth_distances)
        eps = elbow_y
        # Store knee value in dictionary
        knee_values[k] = eps

    return knee_values


def x_stripe_partition_pc(pcd, num_stripes, visualisation=True):
    # Determine the bounds of the point cloud along the x, y, and z axes
    x_min, x_max = pcd.get_min_bound()[0], pcd.get_max_bound()[0]
    y_min, y_max = pcd.get_min_bound()[1], pcd.get_max_bound()[1]
    z_min, z_max = pcd.get_min_bound()[2], pcd.get_max_bound()[2]

    # Compute the width of each stripe
    stripe_width = (x_max - x_min) / num_stripes

    # Split the point cloud into stripes along the x-axis
    stripes = []
    for i in range(num_stripes):
        # Compute the bounding box for the current stripe
        x_start = x_min + i * stripe_width
        x_end = x_start + stripe_width
        y_start, y_end = y_min, y_max
        z_start, z_end = z_min, z_max
        bbox = o3d.geometry.AxisAlignedBoundingBox([x_start, y_start, z_start], [x_end, y_end, z_end])

        # Crop the point cloud to the current stripe using the bounding box
        stripe = pcd.crop(bbox)
        stripes.append(stripe)

    if visualisation:
        # Visualize the results
        o3d.visualization.draw_geometries(stripes)
    return stripes


def y_stripe_partition_pc(pcd, num_stripes, visualisation=True):
    # Determine the bounds of the point cloud along the x, y, and z axes
    x_min, x_max = pcd.get_min_bound()[0], pcd.get_max_bound()[0]
    y_min, y_max = pcd.get_min_bound()[1], pcd.get_max_bound()[1]
    z_min, z_max = pcd.get_min_bound()[2], pcd.get_max_bound()[2]

    # Compute the height of each stripe
    stripe_height = (y_max - y_min) / num_stripes

    # Split the point cloud into stripes along the y-axis
    stripes = []
    for i in range(num_stripes):
        # Compute the bounding box for the current stripe
        x_start, x_end = x_min, x_max
        y_start = y_min + i * stripe_height
        y_end = y_start + stripe_height
        z_start, z_end = z_min, z_max
        bbox = o3d.geometry.AxisAlignedBoundingBox([x_start, y_start, z_start], [x_end, y_end, z_end])

        # Crop the point cloud to the current stripe using the bounding box
        stripe = pcd.crop(bbox)
        stripes.append(stripe)

    if visualisation:
        # Visualize the results
        o3d.visualization.draw_geometries(stripes)
    return stripes


def chunk_partition_pc(point_cloud, visualisation=True, save2image=True):
    ############ Divide the input point cloud into four parts according to the spatial law
    # Get the bounding box of the point cloud
    bbox = point_cloud.get_axis_aligned_bounding_box()

    # Split the bounding box into four parts
    x_mid = (bbox.max_bound[0] + bbox.min_bound[0]) / 2
    y_mid = (bbox.max_bound[1] + bbox.min_bound[1]) / 2

    # Define the four square parts based on the cardinal directions
    bbox_ne = o3d.geometry.AxisAlignedBoundingBox(
        [x_mid, y_mid, bbox.min_bound[2]],
        bbox.max_bound,
    )

    bbox_nw = o3d.geometry.AxisAlignedBoundingBox(
        [bbox.min_bound[0], y_mid, bbox.min_bound[2]],
        [x_mid, bbox.max_bound[1], bbox.max_bound[2]],
    )

    bbox_se = o3d.geometry.AxisAlignedBoundingBox(
        [x_mid, bbox.min_bound[1], bbox.min_bound[2]],
        [bbox.max_bound[0], y_mid, bbox.max_bound[2]],
    )

    bbox_sw = o3d.geometry.AxisAlignedBoundingBox(
        bbox.min_bound,
        [x_mid, y_mid, bbox.max_bound[2]],
    )

    # Extract the points within each bounding box
    pc_ne = point_cloud.crop(bbox_ne)
    pc_nw = point_cloud.crop(bbox_nw)
    pc_se = point_cloud.crop(bbox_se)
    pc_sw = point_cloud.crop(bbox_sw)
    # Verify that the sum of the four tiles equals the number of points in the original point cloud
    assert len(pc_ne.points) + len(pc_nw.points) + len(pc_se.points) + len(pc_sw.points) == len(point_cloud.points)

    if visualisation:
        # Visualize the results
        o3d.visualization.draw_geometries([pc_ne, pc_nw, pc_se, pc_sw])

    if save2image:
        # Create a window and add the point clouds to it
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)  # works for me with False, on some systems needs to be true
        vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
        vis.get_render_option().point_size = 2.0
        vis.add_geometry(pc_ne)
        vis.add_geometry(pc_nw)
        vis.add_geometry(pc_se)
        vis.add_geometry(pc_sw)
        vis.capture_screen_image("result.jpg", do_render=True)

        # Close the window
        vis.destroy_window()

    return pc_ne, pc_nw, pc_se, pc_sw


def get_obb_volume(obb):
    # Compute volume of oriented bounding box
    volume = obb.extent[0] * obb.extent[1] * obb.extent[2]
    return volume


def subdivide_box(box, num_boxes):
    dimensions = box.get_max_bound() - box.get_min_bound()
    aspect_ratio = dimensions / np.max(dimensions)

    centers = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            for k in range(num_boxes):
                center = [
                    box.get_min_bound()[0] + (i + 0.5) * aspect_ratio[0] * dimensions[0],
                    box.get_min_bound()[1] + (j + 0.5) * aspect_ratio[1] * dimensions[1],
                    box.get_min_bound()[2] + (k + 0.5) * aspect_ratio[2] * dimensions[2]
                ]
                centers.append(center)

    boxes = []
    for center in centers:
        min_bound = np.array(center) - aspect_ratio * dimensions / 2
        max_bound = np.array(center) + aspect_ratio * dimensions / 2
        boxes.append(o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound))

    return boxes


def box_refinement(box, low_volume_threshold, high_volume_threshold, height_threshold):
    dimensions = box.get_max_bound() - box.get_min_bound()
    height = dimensions[2]
    width = dimensions[0]
    depth = dimensions[1]
    volume = width * depth * height

    if volume > high_volume_threshold:
        if height >= height_threshold:
            return box
        else:
            num_boxes = int(np.ceil(volume / high_volume_threshold))
            subdivided_boxes = subdivide_box(box, num_boxes)

            new_boxes = []
            for subdivided_box in subdivided_boxes:
                sub_dimensions = subdivided_box.get_max_bound() - subdivided_box.get_min_bound()
                sub_volume = sub_dimensions[0] * sub_dimensions[1] * sub_dimensions[2]
                scale_factor = np.clip(np.power(high_volume_threshold / sub_volume, 1 / 3), 1, None)
                new_width = sub_dimensions[0] / scale_factor
                new_depth = sub_dimensions[1] / scale_factor
                new_height = sub_dimensions[2] / scale_factor
                center = subdivided_box.get_center()
                # min_bound = np.array([center[0] - new_width / 2, center[1] - new_depth / 2, center[2] - new_height / 2])
                # max_bound = np.array([center[0] + new_width / 2, center[1] + new_depth / 2, center[2] + new_height / 2])

                new_box = o3d.geometry.OrientedBoundingBox(center=center, R=np.eye(3),
                                                           extent=np.array([new_width, new_depth, new_height]))
                new_boxes.append(new_box)

            return new_boxes

    elif volume < low_volume_threshold:
        scale_factor = np.power(low_volume_threshold / volume, 1 / 3)
        new_width = width * scale_factor
        new_depth = depth * scale_factor
        new_height = height * scale_factor
        center = box.get_center()
        # min_bound = np.array([center[0] - new_width / 2, center[1] - new_depth / 2, center[2] - new_height / 2])
        # max_bound = np.array([center[0] + new_width / 2, center[1] + new_depth / 2, center[2] + new_height / 2])
        new_box = o3d.geometry.OrientedBoundingBox(center=center, R=np.eye(3),
                                                   extent=np.array([new_width, new_depth, new_height]))
        return new_box

    else:
        return box


def extend_small_box(box):
    center = box.get_center()
    extent = box.extent
    width, depth, height = extent
    if depth < 0.02:
        new_depth = np.random.uniform(0.02, 0.025)
        new_extent = np.array([width, new_depth, height])
        new_box = o3d.geometry.OrientedBoundingBox(center=center, R=box.R, extent=new_extent)
        return new_box
    if width < 0.01:
        new_width = np.random.uniform(0.01, 0.015)
        new_extent = np.array([new_width, depth, height])
        new_box = o3d.geometry.OrientedBoundingBox(center=center, R=box.R, extent=new_extent)
        return new_box


def overlap(box1, box2):
    """Check if two OrientedBoundingBox objects overlap."""
    # Compute the axis-aligned bounding boxes (AABBs) of the oriented bounding boxes
    aabb1 = box1.get_axis_aligned_bounding_box()
    aabb2 = box2.get_axis_aligned_bounding_box()

    # Check for overlap in each dimension (x, y, z)
    for dim in range(3):
        if aabb1.min_bound[dim] > aabb2.max_bound[dim] or aabb1.max_bound[dim] < aabb2.min_bound[dim]:
            return False  # No overlap

    return True  # Overlap exists


def replace_invalid_small_boxes(boxes):
    valid_boxes = []
    invalid_boxes = []

    for box in boxes:
        if box.extent[0] * box.extent[1] > 4.5:
            valid_boxes.append(box)
        else:
            invalid_boxes.append(box)

    for invalid_box in invalid_boxes:
        center = invalid_box.get_center()
        nearest_valid_box = None
        min_distance = np.inf

        for valid_box in valid_boxes:
            valid_center = valid_box.get_center()
            distance = np.linalg.norm(center - valid_center)
            lw_ratio = valid_box.extent[0] / valid_box.extent[1]
            wl_ratio = valid_box.extent[1] / valid_box.extent[0]

            if (1.2 <= lw_ratio <= 3.0 or 0.2 <= wl_ratio <= 0.8) and distance < min_distance:
                nearest_valid_box = valid_box
                min_distance = distance

        if nearest_valid_box is not None:
            # Check for overlap with existing extended boxes
            overlapping_boxes = []
            for box in boxes:
                if box.extent[0] * box.extent[1] > 4.5 and box != nearest_valid_box:
                    if check_overlap(nearest_valid_box, box):
                        overlapping_boxes.append(box)

            if not overlapping_boxes:
                boxes[boxes.index(invalid_box)] = o3d.geometry.OrientedBoundingBox(center=center,
                                                                                   extent=nearest_valid_box.extent,
                                                                                   R=nearest_valid_box.R)
                valid_boxes.remove(nearest_valid_box)  # Remove the used valid box from valid_boxes
            else:
                boxes.remove(invalid_box)  # Remove the invalid box if there is overlap

    return boxes


def check_overlap(box1, box2):
    """Check if two OrientedBoundingBox objects overlap."""
    # Get the corner points of the bounding boxes
    corners1 = np.array(box1.get_box_points())
    corners2 = np.array(box2.get_box_points())

    # Check for overlap along each axis
    for i in range(3):
        min1 = np.min(corners1[:, i])
        max1 = np.max(corners1[:, i])
        min2 = np.min(corners2[:, i])
        max2 = np.max(corners2[:, i])

        if max1 < min2 or min1 > max2:
            return False

    return True


def replace_invalid_small_boxes_v1(boxes):
    valid_boxes = []
    invalid_boxes = []

    for box in boxes:
        if box.extent[0] * box.extent[1] > 4.5:
            valid_boxes.append(box)
        else:
            invalid_boxes.append(box)

    for invalid_box in invalid_boxes:
        center = invalid_box.get_center()
        nearest_valid_box = None
        min_distance = np.inf

        for valid_box in valid_boxes:
            valid_center = valid_box.get_center()
            distance = np.linalg.norm(center - valid_center)
            lw_ratio = valid_box.extent[0] / valid_box.extent[1]
            wl_ratio = valid_box.extent[1] / valid_box.extent[0]

            if (1.2 <= lw_ratio <= 3.0 or 0.2 <= wl_ratio <= 0.8) and distance < min_distance:
                nearest_valid_box = valid_box
                min_distance = distance

        if nearest_valid_box is not None:
            boxes[boxes.index(invalid_box)] = o3d.geometry.OrientedBoundingBox(center=center,
                                                                               extent=nearest_valid_box.extent,
                                                                               R=nearest_valid_box.R)
    return boxes


def replace_invalid_large_boxes(boxes):
    valid_boxes = []
    invalid_boxes = []
    large_boxes = []

    for box in boxes:
        prod = box.extent[0] * box.extent[1]
        if prod > 0.000009 and prod <= 0.000015:
            valid_boxes.append(box)
        elif prod > 0.000015:
            large_boxes.append(box)
        else:
            invalid_boxes.append(box)

    for invalid_box in invalid_boxes:
        boxes.remove(invalid_box)

    for large_box in large_boxes:
        center = large_box.get_center()
        nearest_valid_box = None
        min_distance = np.inf

        for valid_box in valid_boxes:
            valid_center = valid_box.get_center()
            distance = np.linalg.norm(center - valid_center)
            lw_ratio = valid_box.extent[0] / valid_box.extent[1]
            wl_ratio = valid_box.extent[1] / valid_box.extent[0]

            if (2.2 <= lw_ratio <= 2.3 or 0.4 <= wl_ratio <= 0.5) and distance < min_distance:
                nearest_valid_box = valid_box
                min_distance = distance

        if nearest_valid_box is not None:
            extent = nearest_valid_box.extent
            gap = 0.001
            n_copies = int(large_box.extent[0] / extent[0])
            for i in range(n_copies):
                center[0] = nearest_valid_box.get_center()[0] + (i + 1) * extent[0] + gap * i
                boxes.append(o3d.geometry.OrientedBoundingBox(center=center, extent=extent, R=nearest_valid_box.R))

    return boxes


def adjust_box_height(box, min_height=0.007):
    center = box.get_center()
    extent = box.extent
    width, depth, height = extent

    if min_height is not None and height < min_height:
        height = np.random.uniform(0.02, 0.025)
        new_extent = np.array([width, depth, height])
        new_box = o3d.geometry.OrientedBoundingBox(center=center, R=box.R, extent=new_extent)
    # elif max_height is not None and height > max_height:
    #     height = np.random.uniform(0.007, 0.008)
    #     new_extent = np.array([width, depth, height])
    #     new_box = o3d.geometry.OrientedBoundingBox(center=center, R=box.R, extent=new_extent)
    else:
        new_box = o3d.geometry.OrientedBoundingBox(center=center, R=box.R, extent=extent)

    return new_box


def adjust_bounding_box(box, area_threshold=0.0004):
    center = box.get_center()
    extent = box.extent
    width, depth, height = extent

    if height > 0.014:
        return box

    if height >= 0.007 and height <= 0.014:
        top_view_area = width * depth
        if top_view_area > area_threshold:
            new_boxes = divide_large_box(box, width=np.random.uniform(0.01, 0.015),
                                         depth=np.random.uniform(0.02, 0.025), height=np.random.uniform(0.007, 0.008),
                                         gap=0.005, area_threshold=area_threshold)
            return new_boxes

    return box


def divide_large_box(box, width, depth, height, gap, area_threshold):
    center = box.get_center()
    extent = box.extent
    _, _, current_height = extent

    # Calculate the number of boxes needed to split the large box
    top_view_area = extent[0] * extent[1]
    num_boxes = int(np.floor(top_view_area / area_threshold))

    # Calculate the width of each box based on the number of boxes
    box_width = (width + gap) * num_boxes - gap

    # Calculate the start position of the first box
    start_pos = center - np.array([box_width / 2, 0, 0]) + np.array([width / 2, depth / 2, 0])

    # Generate a list of new boxes
    new_boxes = []
    for i in range(num_boxes):
        # Calculate the center position of each box
        box_center = start_pos + np.array([i * (width + gap), 0, 0])

        # Create a new box
        new_extent = np.array([width, depth, height])
        new_box = o3d.geometry.OrientedBoundingBox(center=box_center, R=box.R, extent=new_extent)
        new_boxes.append(new_box)

    return new_boxes


def process_bounding_box(box, area_threshold=0.0003):
    """
    Process a bounding box according to its height and top-down view area.
    If the height is greater than 0.014, do nothing.
    If the height is between 0.007 and 0.014, and the top-down view area is
    greater than area_threshold, divide the box into multiple boxes with
    width between 0.02 and 0.025, height between 0.007 and 0.008, and depth
    between 0.0095 and 0.01, with a gap of 0.01 between neighbouring boxes.
    """
    # Get the height of the bounding box
    _, _, height = box.extent

    if height > 0.014:
        # Do nothing
        return [box]
    elif height >= 0.007:
        # Check the top-down view area
        top_area = box.extent[0] * box.extent[1]
        if top_area > area_threshold:
            # Divide the box
            num_splits = int(np.floor(top_area / area_threshold))
            split_width = (0.025 - 0.02 - (num_splits - 1) * 0.01) / num_splits

            boxes = []
            center = box.get_center()
            R = box.R

            # Iterate over the splits and create new boxes
            for i in range(num_splits):
                extent = [0.02 + i * (split_width + 0.01), 0.01, 0.008]
                new_box = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
                boxes.append(new_box)
                center += R[:, 0] * (extent[0] + 0.01)

            return boxes
        else:
            # Do nothing
            return [box]
    else:
        # Do nothing
        return [box]


def box_list_flatten(box_list):
    flatten_box_list = []
    for box in box_list:
        if isinstance(box, list):
            flatten_box_list.extend(box)
        else:
            flatten_box_list.append(box)
    return flatten_box_list


def find_optimal_dbscan_params(pcd, eps_range, min_points_range):
    """
    Finds the optimal parameters for the DBSCAN algorithm using a grid search.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        The point cloud data to cluster.
    eps_range : list or numpy array
        The range of values for the eps parameter to search over.
    min_points_range : list or numpy array
        The range of values for the min_points parameter to search over.

    Returns
    -------
    tuple
        A tuple containing the optimal eps and min_points values.
    """
    # Convert the Open3D point cloud to a numpy array
    data = np.asarray(pcd.points)

    # Create a grid of parameter combinations
    eps_mesh, min_points_mesh = np.meshgrid(eps_range, min_points_range)
    parameter_combinations = np.column_stack((eps_mesh.ravel(), min_points_mesh.ravel()))

    # Evaluate performance for each parameter combination
    best_score = -1
    best_params = None
    for params in parameter_combinations:
        labels = np.asarray(pcd.cluster_dbscan(eps=params[0], min_points=int(params[1])))
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_params = params

    # Select the best parameter combination
    optimal_eps = best_params[0]
    optimal_min_points = best_params[1]

    return (optimal_eps, optimal_min_points)


def range_based_eps(pcd):
    # Extract x, y, z coordinates
    points = np.asarray(pcd.points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute range along each dimension
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z) - np.min(z)

    # Set epsilon as the maximum range or a fraction of the maximum range
    epsilon = max(x_range, y_range, z_range)

    return epsilon


def density_based_eps(pcd):
    # Extract x, y, z coordinates
    points = np.asarray(pcd.points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute the average point density
    # You can adjust the radius and min_points values for density estimation
    radius = 0.4  # Radius for density estimation
    min_points = 5  # Min points for density estimation
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    densities = []
    for i in range(len(points)):
        [_, indices, _] = kdtree.search_radius_vector_3d(points[i], radius)
        densities.append(len(indices))
    average_density = np.mean(densities)

    # Set epsilon as a fraction of the average density or a fixed value
    epsilon = 0.06 * average_density  # or epsilon = 5
    """

    # Compute k-NN distances for each point
    k = 10  # Number of neighbors to consider
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    knn_dists = []
    for i in range(len(points)):
        [_, indices, _] = kdtree.search_knn_vector_3d(points[i], k)
        knn_dists.append(np.max(np.linalg.norm(points[indices] - points[i], axis=1)))

    # Set adaptive radius as a multiple of the average k-NN distance
    radius_scale = 0.1  # Radius scale factor
    epsilon = radius_scale * np.mean(knn_dists)
    """

    return epsilon


def boxes_postprocessing(boxes):
    updated_boxes = []
    for box in boxes:
        # Check if the box is invalid based on extent area threshold
        if box.extent[0] * box.extent[1] < 5:
            continue

        # Check if the box needs to be split into three smaller boxes
        if box.extent[0] * box.extent[1] > 24:

            # Calculate the dimensions of the split boxes
            split_width = 4.25
            split_length = 2.0
            gap = 0.5

            # Calculate the number of split boxes needed
            num_boxes = int(box.extent[0] * box.extent[1] / (split_width * split_length))

            # Calculate the remaining area after splitting into smaller boxes
            remaining_area = box.extent[0] * box.extent[1] - num_boxes * split_width * split_length

            # Calculate the dimensions of each split box
            split_box_width = split_width
            split_box_length = split_length

            # Check if there's enough remaining area to add another split box
            if remaining_area >= split_box_width * split_box_length:
                num_boxes += 1

            # Calculate the starting point for splitting the box
            start_x = box.center[0] - box.extent[0] / 2.0 + split_box_width / 2.0
            start_y = box.center[1] - box.extent[1] / 2.0 + split_box_length / 2.0

            # Create the split boxes
            for i in range(num_boxes):
                split_box = o3d.geometry.OrientedBoundingBox()
                split_box.center = np.array([start_x + i * (split_box_width + gap), start_y, box.center[2]])
                split_box.extent = np.array([split_box_width, split_box_length, box.extent[2]])
                updated_boxes.append(split_box)
        else:
            # Add the original box to the updated boxes list
            updated_boxes.append(box)

    return updated_boxes


def draw_obbx_from_clusters(pcd, epsilon, min_points_box, visual_clusters=True, visual_per_box=True,
                            visual_all_boxes=True, visual_covered_points=True):
    cluster_indices = np.asarray(pcd.cluster_dbscan(eps=epsilon, min_points=10))
    # Count the number of points in each cluster
    for i in set(cluster_indices):
        print("Cluster", i + 1, "has", np.count_nonzero(cluster_indices == i), "points.")
    # Count the number of clusters
    max_label = cluster_indices.max()
    print(f"point cloud has {max_label + 1} clusters")

    if visual_clusters:
        # Create a list of colors for each cluster
        colors = np.random.uniform(0, 1, size=(max_label + 1, 3))
        # Set the first color to black
        colors[0] = [0, 0, 0]
        # Assign colors to each point based on their cluster index
        pcd.colors = o3d.utility.Vector3dVector(colors[cluster_indices])
        # o3d.visualization.draw_geometries([pcd])

    ########## Create an empty list to store the OBBs
    boxes = []
    box_centres = []
    box_dims = []
    covered_points = []
    for i in range(max_label):
        # Create a new point cloud from the cluster centers
        indices = np.where(cluster_indices == i)[0]
        cluster_points = np.asarray(pcd.points)[indices]
        # Specify the threshold value
        if cluster_points.shape[0] >= min_points_box:
            ######### Create a point cloud object
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points.reshape(-1, 3))
            box = cluster_pcd.get_oriented_bounding_box()
            box.color = (0.6, 0.6, 0.6)
            # print(f"the {i}th box's has x:{box.extent[0]}------y:{box.extent[1]}--------z:{box.extent[2]}")
            #############  cars wl_area = 8, vans wl_area = 13.67, three cars = 25.8, two cars = 17-19
            x, y, z = box.extent
            if x * y >= 5.5 and x / y > 1.3 and y < 3.8:
                boxes.append(box)
                box_centres.append(box.center)
                box_dims.append(box.get_max_bound() - box.get_min_bound())
                covered_points.extend(cluster_points)
            if visual_per_box:
                o3d.visualization.draw_geometries([cluster_pcd, box])

    if visual_all_boxes:
        o3d.visualization.draw_geometries([pcd] + boxes)

    if visual_covered_points:
        # Create a new point cloud from the covered points
        covered_pcd = o3d.geometry.PointCloud()
        covered_pcd.points = o3d.utility.Vector3dVector(np.asarray(covered_points))
        # Create a list of colors for each cluster
        colors = np.random.uniform(0, 1, size=(max_label + 1, 3))
        # Set the first color to black
        colors[0] = [0, 0, 0]
        # Assign colors to each point based on their cluster index
        covered_pcd.colors = o3d.utility.Vector3dVector(colors[cluster_indices])
        o3d.visualization.draw_geometries([covered_pcd] + boxes)

    return boxes, box_centres, box_dims, covered_points


def mean_shift_clustering(pcd, bandwidth=0.1, radius=0.5, max_nn=100):
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)

    # Run mean shift clustering
    labels = np.zeros(points.shape[0])
    cluster_idx = 0
    for i in range(points.shape[0]):
        if labels[i] == 0:
            seed = [i]
            cluster = []
            while len(seed) > 0:
                idx = seed.pop(0)
                if labels[idx] == 0:
                    distances = np.linalg.norm(points[idx] - points, axis=1)
                    neighbors = np.where(distances < radius)[0]
                    if len(neighbors) < max_nn:
                        cluster.append(idx)
                        labels[idx] = cluster_idx
                        seed.extend(neighbors)
            cluster_idx += 1

    # Visualize clusters and bounding boxes
    unique_labels = np.unique(labels)
    bbox_list = []
    for i, label in enumerate(unique_labels):
        indices = np.where(labels == i)[0]
        cluster_points = np.asarray(pcd.points)[indices]
        # Specify the threshold value
        if cluster_points.shape[0] >= 10:
            ######### Create a point cloud object
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points.reshape(-1, 3))
            obbox = cluster_pcd.get_oriented_bounding_box(robust=True)
            obbox.color = (0.6, 0.6, 0.6)
            bbox_list.append(obbox)
        else:
            continue
    o3d.visualization.draw_geometries([pcd] + bbox_list)
    return labels


def spectral_clustering(pcd, n_clusters=5, gamma=10):
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)
    # Compute feature descriptors (normal vectors) for each point
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)

    # Concatenate point coordinates and normal vectors as feature descriptors
    features = np.hstack((points, normals))
    # Set the number of clusters and create spectral clustering object
    sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=50)
    # Fit the spectral clustering object to the feature descriptors
    labels = sc.fit_predict(features)
    # Visualize clusters and bounding boxes
    unique_labels = np.unique(labels)
    bbox_list = []
    for i, label in enumerate(unique_labels):
        indices = np.where(labels == i)[0]
        cluster_points = np.asarray(pcd.points)[indices]
        # Specify the threshold value
        if cluster_points.shape[0] >= 10:
            ######### Create a point cloud object
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points.reshape(-1, 3))
            obbox = cluster_pcd.get_oriented_bounding_box(robust=True)
            obbox.color = (0.6, 0.6, 0.6)
            bbox_list.append(obbox)
        else:
            continue
    o3d.visualization.draw_geometries([pcd] + bbox_list)
    return labels
