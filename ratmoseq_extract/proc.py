"""
Video pre-processing utilities for detecting ROIs and extracting raw data.
"""

import cv2
import joblib
import tarfile
import scipy.stats
import numpy as np
import scipy.signal
import skimage.measure
import scipy.interpolate
import skimage.morphology
from copy import deepcopy
from tqdm.auto import tqdm
import warnings
from os.path import exists, join, dirname
from ratmoseq_extract.io import read_image, write_image, load_movie_data, get_movie_info


def plane_fit3(points):
    """
    Fit a plane to 3 points (min number of points for fitting a plane)

    Args:
    points (numpy.ndarray): each row is a group of points, columns correspond to x,y,z.

    Returns:
    plane (numpy.array): linear plane fit-->a*x+b*y+c*z+d
    """

    a = points[1] - points[0]
    b = points[2] - points[0]
    # cross prod to make sure the three points make an area, hence a plane.
    normal = np.array(
        [
            [a[1] * b[2] - a[2] * b[1]],
            [a[2] * b[0] - a[0] * b[2]],
            [a[0] * b[1] - a[1] * b[0]],
        ]
    ).astype("float")
    denom = np.sum(np.square(normal)).astype("float")
    if denom < np.spacing(1):
        plane = np.empty((4,))
        plane[:] = np.nan
    else:
        normal /= np.sqrt(denom)
        d = np.dot(-points[0], normal)
        plane = np.hstack((normal.flatten(), d))

    return plane


def plane_ransac(
    depth_image,
    bg_roi_depth_range=(900, 1000),
    iters=1000,
    noise_tolerance=30,
    in_ratio=0.1,
    progress_bar=False,
    mask=None,
    **kwargs,
):
    """
    Fit a plane using a naive RANSAC implementation

    Args:
    depth_image (numpy.ndarray): background image to fit plane to
    bg_roi_depth_range (tuple): min/max depth (mm) to consider pixels for plane
    iters (int): number of RANSAC iterations
    noise_tolerance (float): distance from plane to consider a point an inlier
    in_ratio (float): fraction of points required to consider a plane fit good
    progress_bar (bool): display progress bar
    mask (numpy.ndarray): boolean mask to find region to use
    kwargs (dict): dictionary containing extra keyword arguments from moseq2_extract.proc.get_roi()

    Returns:
    best_plane (numpy.array): plane fit to data
    dist (numpy.array): distance of the calculated coordinates and "best plane"
    """

    use_points = np.logical_and(
        depth_image > bg_roi_depth_range[0], depth_image < bg_roi_depth_range[1]
    )
    if np.sum(use_points) <= 10:
        raise ValueError(
            f'Too few datapoints exist within given "bg roi depth range" {bg_roi_depth_range} -- data point count: {np.sum(use_points)}.'
            "Please adjust this parameter to fit your recording sessions."
        )

    if mask is not None:
        use_points = np.logical_and(use_points, mask)

    xx, yy = np.meshgrid(
        np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0])
    )

    coords = np.vstack(
        (
            xx[use_points].ravel(),
            yy[use_points].ravel(),
            depth_image[use_points].ravel(),
        )
    )
    coords = coords.T

    best_dist = np.inf
    best_num = 0

    npoints = np.sum(use_points)

    for _ in tqdm(range(iters), disable=not progress_bar, desc="Finding plane"):

        sel = coords[np.random.choice(coords.shape[0], 3, replace=True)]
        tmp_plane = plane_fit3(sel)

        if np.all(np.isnan(tmp_plane)):
            continue

        dist = np.abs(np.dot(coords, tmp_plane[:3]) + tmp_plane[3])
        inliers = dist < noise_tolerance
        ninliers = np.sum(inliers)

        if (
            (ninliers / npoints) > in_ratio
            and ninliers > best_num
            and np.mean(dist) < best_dist
        ):
            best_dist = np.mean(dist)
            best_num = ninliers
            best_plane = tmp_plane

    # fit the plane to our x,y,z coordinates
    coords = np.vstack((xx.ravel(), yy.ravel(), depth_image.ravel())).T
    dist = np.abs(np.dot(coords, best_plane[:3]) + best_plane[3])

    return best_plane, dist


def compute_plane_bground(frames_file, finfo, bg_roi_depth_range=(900, 1000), **kwargs):
    """
    Compute plane background image from video file.

    Args:
    frame_shape (numpy.ndarray): shape of the frame
    plane (numpy.ndarray): plane parameters

    Returns:
    plane_im (numpy.ndarray): plane background image
    """

    depth_image = load_movie_data(
        frames_file,
        0,
        frame_size=finfo["dims"],
        finfo=finfo,
        **kwargs,
    ).squeeze()
    frame_shape = depth_image.shape

    plane, _ = plane_ransac(depth_image, bg_roi_depth_range, **kwargs)

    xx, yy = np.meshgrid(
        np.arange(frame_shape.shape[1]), np.arange(frame_shape.shape[0])
    )
    coords = np.vstack((xx.ravel(), yy.ravel()))

    plane_im = (np.dot(coords.T, plane[:2]) + plane[3]) / -plane[2]
    plane_im = plane_im.reshape(frame_shape)

    return plane_im, depth_image


def compute_median_bground(
    frames_file, frame_stride=500, med_scale=5, finfo=None, **kwargs
):
    """
    Compute median background image from video file.

    Args:
    frames_file (str): path to the depth video
    frame_stride (int): stride size between frames for median bground calculation
    med_scale (int): kernel size for median blur for background images.
    kwargs (dict): extra keyword arguments

    Returns:
    bground (numpy.ndarray): background
    frame_store[0] (numpy.ndarray): first frame
    """

    frame_idx = np.arange(0, finfo["nframes"], frame_stride)
    frame_store = []
    for i, frame in enumerate(frame_idx):
        frs = load_movie_data(
            frames_file, [int(frame)], frame_size=finfo["dims"], finfo=finfo, **kwargs
        ).squeeze()
        frame_store.append(cv2.medianBlur(frs, med_scale))

    bground = np.nanmedian(frame_store, axis=0)

    return bground, frame_store[0]


def get_bground(
    frames_file,
    bground_type="median",
    frame_stride=500,
    med_scale=5,
    bg_roi_depth_range=(900, 1000),
    output_dir=None,
    **kwargs,
):
    """
    Compute median or plane background image from video file.

    Args:
    frames_file (str): path to the depth video
    bground_type (str): type of background to compute
    frame_stride (int): stride size between frames for median bground calculation
    med_scale (int): kernel size for median blur for background images.
    output_dir (str): output directory to save background image
    kwargs (dict): extra keyword arguments

    Returns:
    bground (numpy.ndarray): background image
    first_frame (numpy.ndarray): first frame of video
    """

    if output_dir is None:
        bground_path = join(dirname(frames_file), "proc", "bground.tiff")
    else:
        bground_path = join(output_dir, "bground.tiff")

    finfo = get_movie_info(frames_file, **kwargs)
    if bground_type == "median":
        bground, first_frame = compute_median_bground(
            frames_file, frame_stride, med_scale, finfo, **kwargs
        )
        write_image(bground_path, bground, scale=True)
    else:
        plane, _ = plane_ransac(finfo["dims"], **kwargs)
        bground, first_frame = compute_plane_bground(
            frames_file, finfo, bg_roi_depth_range, **kwargs
        )

    write_image(bground_path, bground, scale=True)

    return bground, first_frame


def get_strels(config_data):
    """
    Get dictionary object of cv2 StructuringElements for image filtering given
    a dict of configurations parameters.

    Args:
    config_data (dict): dict containing cv2 Structuring Element parameters

    Returns:
    str_els (dict): dict containing cv2 StructuringElements used for image filtering
    """

    str_els = {
        "strel_dilate": select_strel(
            config_data["bg_roi_shape"], tuple(config_data["bg_roi_dilate"])
        ),
        "strel_erode": select_strel(
            config_data["bg_roi_shape"], tuple(config_data["bg_roi_erode"])
        ),
        "strel_tail": select_strel(
            config_data["tail_filter_shape"], tuple(config_data["tail_filter_size"])
        ),
        "strel_min": select_strel(
            config_data["cable_filter_shape"], tuple(config_data["cable_filter_size"])
        ),
    }

    return str_els


def select_strel(string="e", size=(10, 10)):
    """
    Returns structuring element of specified shape.

    Args:
    string (str): string to indicate whether to use ellipse or rectangle
    size (tuple): size of structuring element

    Returns:
    strel (cv2.StructuringElement): selected cv2 StructuringElement to use in video filtering or ROI dilation/erosion.
    """

    if string[0].lower() == "e":
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    elif string[0].lower() == "r":
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    else:
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    return strel


def check_filter_sizes(config_data):
    """
    Ensure spatial and temporal filter kernel sizes are odd numbers.

    Args:
    config_data (dict): a dictionary holding all extraction parameters

    Returns:
    config_data (dict): Updated configuration dict

    """

    # Ensure filter kernel sizes are odd
    if (
        config_data["spatial_filter_size"][0] % 2 == 0
        and config_data["spatial_filter_size"][0] > 0
    ):
        warnings.warn(
            "Spatial Filter Size must be an odd number. Incrementing value by 1."
        )
        config_data["spatial_filter_size"][0] += 1
    if (
        config_data["temporal_filter_size"][0] % 2 == 0
        and config_data["temporal_filter_size"][0] > 0
    ):
        config_data["temporal_filter_size"][0] += 1
        warnings.warn(
            "Spatial Filter Size must be an odd number. Incrementing value by 1."
        )

    return config_data


def get_flips(frames, flip_file=None, smoothing=None):
    """
    Predict frames where mouse orientation is flipped to later correct.

    Args:
    frames (numpy.ndarray): frames x rows x columns, cropped mouse
    flip_file (str): path to pre-trained scipy random forest classifier
    smoothing (int): kernel size for median filter smoothing of random forest probabilities

    Returns:
    flips (numpy.array):  array for flips
    """

    try:
        clf = joblib.load(flip_file)
    except IOError:
        print(f"Could not open file {flip_file}")
        raise

    flip_class = np.where(clf.classes_ == 1)[0]

    try:
        probas = clf.predict_proba(
            frames.reshape((-1, frames.shape[1] * frames.shape[2]))
        )
    except ValueError:
        if (
            hasattr(clf, "n_features_")
            and int(np.sqrt(clf.n_features_)) != frames.shape[-1]
        ):
            print("WARNING: Input crop-size is not compatible with flip classifier.")
            accepted_crop = int(np.sqrt(clf.n_features_))
            print(
                f"Adjust the crop-size to ({accepted_crop}, {accepted_crop}) to use this flip classifier."
            )
        print("Frames shape:", frames.shape)
        print("The extracted data will NOT be flipped!")
        probas = np.array(
            [[0] * len(frames), [1] * len(frames)]
        ).T  # default output; indicating no flips

    if smoothing:
        for i in range(probas.shape[1]):
            probas[:, i] = scipy.signal.medfilt(probas[:, i], smoothing)

    flips = probas.argmax(axis=1) == flip_class

    return flips


def get_largest_cc(frames, progress_bar=False):
    """
    Returns largest connected component blob in image

    Args:
    frames (numpy.ndarray): frames x rows x columns, uncropped mouse
    progress_bar (bool): display progress bar

    Returns:
    foreground_obj (numpy.ndarray):  frames x rows x columns, true where blob was found
    """

    foreground_obj = np.zeros((frames.shape), "bool")

    for i in tqdm(
        range(frames.shape[0]),
        disable=not progress_bar,
        desc="Computing largest Connected Component",
    ):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            frames[i], connectivity=4
        )
        szs = stats[:, -1]
        foreground_obj[i] = output == szs[1:].argmax() + 1

    return foreground_obj


def get_bbox(roi):
    """
    return an array with the x and y boundaries given ROI.

    Args:
    roi (np.ndarray): ROI boolean mask to calculate bounding box.

    Returns:
    bbox (np.ndarray): Bounding Box around ROI
    """

    y, x = np.where(roi > 0)

    if len(y) == 0 or len(x) == 0:
        return None
    else:
        bbox = np.array([[y.min(), x.min()], [y.max(), x.max()]])
        return bbox


def threshold_chunk(chunk, min_height, max_height):
    """
    Threshold out depth values that are less than min_height and larger than
    max_height.

    Args:
    chunk (np.ndarray): Chunk of frames to threshold (nframes, width, height)
    min_height (int): Minimum depth values to include after thresholding.
    max_height (int): Maximum depth values to include after thresholding.
    dilate_iterations (int): Number of iterations the ROI was dilated.

    Returns:
    chunk (3D np.ndarray): Updated frame chunk.
    """

    chunk[chunk < min_height] = 0
    chunk[chunk > max_height] = 0

    return chunk


def im_moment_features(IM):
    """
    Use the method of moments and centralized moments to get image properties.

    Args:
    IM (numpy.ndarray): depth image

    Returns:
    features (dict): returns a dictionary with orientation, centroid, and ellipse axis length
    """

    tmp = cv2.moments(IM)
    num = 2 * tmp["mu11"]
    den = tmp["mu20"] - tmp["mu02"]

    common = np.sqrt(4 * np.square(tmp["mu11"]) + np.square(den))

    if tmp["m00"] == 0:
        features = {
            "orientation": np.nan,
            "centroid": np.nan,
            "axis_length": [np.nan, np.nan],
        }
    else:
        features = {
            "orientation": -0.5 * np.arctan2(num, den),
            "centroid": [tmp["m10"] / tmp["m00"], tmp["m01"] / tmp["m00"]],
            "axis_length": [
                2
                * np.sqrt(2)
                * np.sqrt((tmp["mu20"] + tmp["mu02"] + common) / tmp["m00"]),
                2
                * np.sqrt(2)
                * np.sqrt((tmp["mu20"] + tmp["mu02"] - common) / tmp["m00"]),
            ],
        }

    return features


def clean_frames(
    frames,
    prefilter_space=(3,),
    prefilter_time=None,
    strel_tail=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    iters_tail=None,
    strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    iters_min=None,
    progress_bar=False,
):
    """
    Simple temporal and/or spatial filtering, median filter and morphological opening.

    Args:
    frames (np.ndarray): Frames (frames x rows x columns) to filter.
    prefilter_space (tuple): kernel size for spatial filtering
    prefilter_time (tuple): kernel size for temporal filtering
    strel_tail (cv2.StructuringElement): Element for tail filtering.
    iters_tail (int): number of iterations to run opening
    frame_dtype (str): frame encodings
    strel_min (int): minimum kernel size
    iters_min (int): minimum number of filtering iterations
    progress_bar (bool): display progress bar

    Returns:
    filtered_frames (numpy.ndarray): frames x rows x columns
    """

    # seeing enormous speed gains w/ opencv
    filtered_frames = frames.copy()

    for i in tqdm(
        range(frames.shape[0]), disable=not progress_bar, desc="Cleaning frames"
    ):
        # Erode Frames
        if iters_min is not None and iters_min > 0:
            filtered_frames[i] = cv2.erode(filtered_frames[i], strel_min, iters_min)
        # Median Blur
        if prefilter_space is not None and np.all(np.array(prefilter_space) > 0):
            for j in range(len(prefilter_space)):
                filtered_frames[i] = cv2.medianBlur(
                    filtered_frames[i], prefilter_space[j]
                )
        # Tail Filter
        if iters_tail is not None and iters_tail > 0:
            filtered_frames[i] = cv2.morphologyEx(
                filtered_frames[i], cv2.MORPH_OPEN, strel_tail, iters_tail
            )

    # Temporal Median Filter
    if prefilter_time is not None and np.all(np.array(prefilter_time) > 0):
        for j in range(len(prefilter_time)):
            filtered_frames = scipy.signal.medfilt(
                filtered_frames, [prefilter_time[j], 1, 1]
            )

    return filtered_frames


def get_frame_features(frames, frame_threshold=10, use_cc=False, progress_bar=False):
    """
    Use image moments to compute features of the largest object in the frame

    Args:
    frames (3d np.ndarray): input frames
    frame_threshold (int): threshold in mm separating floor from mouse
    mask (3d np.ndarray): input frame mask for parts not to filter.
    mask_threshold (int): threshold to include regions into mask.
    use_cc (bool): Use connected components.
    progress_bar (bool): Display progress bar.

    Returns:
    features (dict of lists): dictionary with simple image features
    mask (3d np.ndarray): input frame mask.
    """

    nframes = frames.shape[0]

    # Get frame mask
    if type(mask) is np.ndarray and mask.size > 0:
        has_mask = True
    else:
        has_mask = False
        mask = np.zeros((frames.shape), "uint8")

    # Pack contour features into dict
    features = {
        "centroid": np.full((nframes, 2), np.nan),
        "orientation": np.full((nframes,), np.nan),
        "axis_length": np.full((nframes, 2), np.nan),
    }

    for i in tqdm(range(nframes), disable=not progress_bar, desc="Computing moments"):
        # Threshold frame to compute mask
        frame_mask = frames[i] > frame_threshold

        # Get contours in frame
        cnts, hierarchy = cv2.findContours(
            frame_mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        tmp = np.array([cv2.contourArea(x) for x in cnts])

        if tmp.size == 0:
            continue

        mouse_cnt = tmp.argmax()

        # Get features from contours
        for key, value in im_moment_features(cnts[mouse_cnt]).items():
            features[key][i] = value

    return features, mask


def crop_and_rotate_frames(frames, features, crop_size=(80, 80), progress_bar=False):
    """
    Crop mouse from image and orients it such that the head is pointing right

    Args:
    frames (3d np.ndarray): frames to crop and rotate
    features (dict): dict of extracted features, found in result_00.h5 files.
    crop_size (tuple): size of cropped image.
    progress_bar (bool): Display progress bar.

    Returns:
    cropped_frames (3d np.ndarray): Crop and rotated frames.
    """

    nframes = frames.shape[0]

    # Prepare cropped frame array
    cropped_frames = np.zeros((nframes, crop_size[0], crop_size[1]), frames.dtype)

    # Get window dimensions
    win = (crop_size[0] // 2, crop_size[1] // 2 + 1)
    border = (crop_size[1], crop_size[1], crop_size[0], crop_size[0])

    for i in tqdm(range(frames.shape[0]), disable=not progress_bar, desc="Rotating"):

        if np.any(np.isnan(features["centroid"][i])):
            continue

        # Get bounded frames
        use_frame = cv2.copyMakeBorder(frames[i], *border, cv2.BORDER_CONSTANT, 0)

        # Get row and column centroids
        rr = np.arange(
            features["centroid"][i, 1] - win[0], features["centroid"][i, 1] + win[1]
        ).astype("int16")
        cc = np.arange(
            features["centroid"][i, 0] - win[0], features["centroid"][i, 0] + win[1]
        ).astype("int16")

        rr = rr + crop_size[0]
        cc = cc + crop_size[1]

        # Ensure centroids are in bounded frame
        if (
            np.any(rr >= use_frame.shape[0])
            or np.any(rr < 1)
            or np.any(cc >= use_frame.shape[1])
            or np.any(cc < 1)
        ):
            continue

        # Rotate the frame such that the mouse is oriented facing east
        rot_mat = cv2.getRotationMatrix2D(
            (crop_size[0] // 2, crop_size[1] // 2),
            -np.rad2deg(features["orientation"][i]),
            1,
        )
        cropped_frames[i] = cv2.warpAffine(
            use_frame[rr[0] : rr[-1], cc[0] : cc[-1]],
            rot_mat,
            (crop_size[0], crop_size[1]),
        )

    return cropped_frames


def convert_pxs_to_mm(
    coords, resolution=(512, 424), field_of_view=(70.6, 60), true_depth=673.1
):
    """
    Converts x, y coordinates in pixel space to mm.

    Args:
    coords (list): list of x,y pixel coordinates
    resolution (tuple): image dimensions
    field_of_view (tuple): width and height scaling params
    true_depth (float): detected true depth

    Returns:
    new_coords (list): x,y coordinates in mm
    """

    # http://stackoverflow.com/questions/17832238/kinect-intrinsic-parameters-from-field-of-view/18199938#18199938
    # http://www.imaginativeuniversal.com/blog/post/2014/03/05/quick-reference-kinect-1-vs-kinect-2.aspx
    # http://smeenk.com/kinect-field-of-view-comparison/

    cx = resolution[0] // 2
    cy = resolution[1] // 2

    xhat = coords[:, 0] - cx
    yhat = coords[:, 1] - cy

    fw = resolution[0] / (2 * np.deg2rad(field_of_view[0] / 2))
    fh = resolution[1] / (2 * np.deg2rad(field_of_view[1] / 2))

    new_coords = np.zeros_like(coords)
    new_coords[:, 0] = true_depth * xhat / fw
    new_coords[:, 1] = true_depth * yhat / fh

    return new_coords


def compute_scalars(
    frames, track_features, min_height=10, max_height=100, true_depth=673.1
):
    """
    Compute extracted scalars.

    Args:
    frames (np.ndarray): frames x r x c, uncropped mouse
    track_features (dict):  dictionary with tracking variables (centroid and orientation)
    min_height (float): minimum height of the mouse
    max_height (float): maximum height of the mouse
    true_depth (float): detected true depth

    Returns:
    features (dict): dictionary of scalars
    """

    nframes = frames.shape[0]

    # Pack features into dict
    features = {
        "centroid_x_px": np.zeros((nframes,), "float32"),
        "centroid_y_px": np.zeros((nframes,), "float32"),
        "velocity_2d_px": np.zeros((nframes,), "float32"),
        "velocity_3d_px": np.zeros((nframes,), "float32"),
        "width_px": np.zeros((nframes,), "float32"),
        "length_px": np.zeros((nframes,), "float32"),
        "area_px": np.zeros((nframes,)),
        "centroid_x_mm": np.zeros((nframes,), "float32"),
        "centroid_y_mm": np.zeros((nframes,), "float32"),
        "velocity_2d_mm": np.zeros((nframes,), "float32"),
        "velocity_3d_mm": np.zeros((nframes,), "float32"),
        "width_mm": np.zeros((nframes,), "float32"),
        "length_mm": np.zeros((nframes,), "float32"),
        "area_mm": np.zeros((nframes,)),
        "height_ave_mm": np.zeros((nframes,), "float32"),
        "angle": np.zeros((nframes,), "float32"),
        "velocity_theta": np.zeros((nframes,)),
    }

    # Get mm centroid
    centroid_mm = convert_pxs_to_mm(track_features["centroid"], true_depth=true_depth)
    centroid_mm_shift = convert_pxs_to_mm(
        track_features["centroid"] + 1, true_depth=true_depth
    )

    # Based on the centroid of the mouse, get the mm_to_px conversion
    px_to_mm = np.abs(centroid_mm_shift - centroid_mm)
    masked_frames = np.logical_and(frames > min_height, frames < max_height)

    features["centroid_x_px"] = track_features["centroid"][:, 0]
    features["centroid_y_px"] = track_features["centroid"][:, 1]

    features["centroid_x_mm"] = centroid_mm[:, 0]
    features["centroid_y_mm"] = centroid_mm[:, 1]

    # based on the centroid of the mouse, get the mm_to_px conversion

    features["width_px"] = np.min(track_features["axis_length"], axis=1)
    features["length_px"] = np.max(track_features["axis_length"], axis=1)
    features["area_px"] = np.sum(masked_frames, axis=(1, 2))

    features["width_mm"] = features["width_px"] * px_to_mm[:, 1]
    features["length_mm"] = features["length_px"] * px_to_mm[:, 0]
    features["area_mm"] = features["area_px"] * px_to_mm.mean(axis=1)

    features["angle"] = track_features["orientation"]

    nmask = np.sum(masked_frames, axis=(1, 2))

    for i in range(nframes):
        if nmask[i] > 0:
            features["height_ave_mm"][i] = np.mean(frames[i, masked_frames[i]])

    vel_x = np.diff(
        np.concatenate((features["centroid_x_px"][:1], features["centroid_x_px"]))
    )
    vel_y = np.diff(
        np.concatenate((features["centroid_y_px"][:1], features["centroid_y_px"]))
    )
    vel_z = np.diff(
        np.concatenate((features["height_ave_mm"][:1], features["height_ave_mm"]))
    )

    features["velocity_2d_px"] = np.hypot(vel_x, vel_y)
    features["velocity_3d_px"] = np.sqrt(
        np.square(vel_x) + np.square(vel_y) + np.square(vel_z)
    )

    vel_x = np.diff(
        np.concatenate((features["centroid_x_mm"][:1], features["centroid_x_mm"]))
    )
    vel_y = np.diff(
        np.concatenate((features["centroid_y_mm"][:1], features["centroid_y_mm"]))
    )

    features["velocity_2d_mm"] = np.hypot(vel_x, vel_y)
    features["velocity_3d_mm"] = np.sqrt(
        np.square(vel_x) + np.square(vel_y) + np.square(vel_z)
    )

    features["velocity_theta"] = np.arctan2(vel_y, vel_x)

    return features


def model_smoother(features, ll=None, clips=(-300, -125)):
    """
    Apply spatial feature filtering.

    Args:
    features (dict): dictionary of extraction scalar features
    ll (numpy.array): array of loglikelihoods of pixels in frame
    clips (tuple): tuple to ensure video is indexed properly

    Returns:
    features (dict): smoothed version of input features
    """

    if ll is None or clips is None or (clips[0] >= clips[1]):
        return features

    ave_ll = np.zeros((ll.shape[0],))
    for i, ll_frame in enumerate(ll):

        max_mu = clips[1]
        min_mu = clips[0]

        smoother = np.mean(ll[i])
        smoother -= min_mu
        smoother /= max_mu - min_mu

        smoother = np.clip(smoother, 0, 1)
        ave_ll[i] = smoother

    for k, v in features.items():
        nans = np.isnan(v)
        ndims = len(v.shape)
        xvec = np.arange(len(v))
        if nans.any():
            if ndims == 2:
                for i in range(v.shape[1]):
                    f = scipy.interpolate.interp1d(
                        xvec[~nans[:, i]],
                        v[~nans[:, i], i],
                        kind="nearest",
                        fill_value="extrapolate",
                    )
                    fill_vals = f(xvec[nans[:, i]])
                    features[k][xvec[nans[:, i]], i] = fill_vals
            else:
                f = scipy.interpolate.interp1d(
                    xvec[~nans], v[~nans], kind="nearest", fill_value="extrapolate"
                )
                fill_vals = f(xvec[nans])
                features[k][nans] = fill_vals

    for i in range(2, len(ave_ll)):
        smoother = ave_ll[i]
        for k, v in features.items():
            features[k][i] = (1 - smoother) * v[i - 1] + smoother * v[i]

    for i in reversed(range(len(ave_ll) - 1)):
        smoother = ave_ll[i]
        for k, v in features.items():
            features[k][i] = (1 - smoother) * v[i + 1] + smoother * v[i]

    return features
