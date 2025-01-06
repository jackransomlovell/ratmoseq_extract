"""
Extraction helper utility for computing scalar feature values performing cleaning, cropping and rotating operations.
"""

import cv2
import numpy as np
from copy import deepcopy
from ratmoseq_extract.sam2 import get_sam2_predictor, segment_chunk
from ratmoseq_extract.proc import (
    crop_and_rotate_frames,
    threshold_chunk,
    clean_frames,
    apply_roi,
    get_frame_features,
    get_flips,
    compute_scalars
)


# one stop shopping for taking some frames and doing stuff
def extract_chunk(
    chunk,
    spatial_filter_size=(3,),
    temporal_filter_size=None,
    tail_filter_iters=1,
    iters_min=0,
    strel_tail=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    strel_min=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    min_height=10,
    max_height=100,
    mask_threshold=-20,
    use_cc=False,
    bground=None,
    roi=None,
    flip_classifier=None,
    flip_classifier_smoothing=51,
    progress_bar=True,
    crop_size=(256, 256),
    true_depth=950,
    compute_raw_scalars=False,
    sam2=False,
    sam2_checkpoint=None,
    sam2_points=None,
    **kwargs
):
    """
    Extract mouse from the depth videos.

    Args:
    chunk (np.ndarray): chunk to extract - (chunksize, height, width)
    use_tracking_model (bool): The EM tracker uses expectation-maximization to fit improve mouse detection.
    spatial_filter_size (tuple): spatial kernel size used in median filtering.
    temporal_filter_size (tuple): temporal kernel size used in median filtering.
    tail_filter_iters (int): number of filtering iterations on mouse tail
    iters_min (int): minimum tail filtering filter kernel size
    strel_tail (cv2::StructuringElement): filtering kernel size to filter out mouse tail.
    strel_min (cv2::StructuringElement): filtering kernel size to filter mouse body in cable recording cases.
    min_height (int): minimum (mm) distance of mouse to floor.
    max_height (int): maximum (mm) distance of mouse to floor.
    mask_threshold (int): Threshold on log-likelihood to include pixels for centroid and angle calculation
    use_cc (bool): boolean to use connected components in cv2 structuring elements
    bground (np.ndarray): 2D numpy array representing previously computed median background image of entire extracted recording.
    roi (np.ndarray): 2D numpy array representing previously computed roi (area of bucket floor) to search for mouse within.
    flip_classifier (str): path to pre-selected flip classifier.
    flip_classifier_smoothing (int): amount of smoothing to use for flip classifier.
    save_path: (str): Path to save extracted results
    progress_bar (bool): Display progress bar
    crop_size (tuple): size of the cropped mouse image.
    true_depth (float): the computed detected true depth value for the middle of the arena
    model_smoothing_clips (tuple): Model smoothing clips
    tracking_model_init (str): Method for tracking model initialization
    compute_raw_scalars (bool): Compute scalars from unfiltered crop-rotated data.

    Returns:
    results (dict): dict object containing the following keys:
    chunk (numpy.ndarray): bg subtracted and applied ROI version of original video chunk
    depth_frames(numpy.ndarray): cropped and oriented mouse video chunk
    mask_frames (numpy.ndarray): cropped and oriented mouse video chunk
    scalars (dict): computed scalars (str) mapped to 1d numpy arrays of length=nframes.
    flips(1d array): list of frame indices where the mouse orientation was flipped.
    parameters (dict): mean and covariance estimates for each frame (if em_tracking=True), otherwise None.
    """

    if bground is not None:
        # Perform background subtraction
        if not kwargs.get('graduate_walls', False):
            # pixels with a value of 0 are depth values that could not be computed
            chunk = ((bground - chunk) * (chunk != 0)).astype(int)
        else:
            # Subtracting only background area where mouse is not on the bucket edge
            mouse_on_edge = (bground < true_depth) & (chunk < bground)
            chunk = (bground - chunk) * np.logical_not(mouse_on_edge) + (
                true_depth - chunk
            ) * mouse_on_edge

        # Threshold chunk depth values at min and max heights
        chunk = threshold_chunk(chunk, min_height, max_height).astype(int)

    # Apply ROI mask
    if roi is not None:
        chunk = apply_roi(chunk, roi)
        # TODO: modify the keypoint coords to reflect the new ROI
    
    # pack clean params into a dict
    clean_params = {
        'prefilter_space': spatial_filter_size,
        'prefilter_time': temporal_filter_size,
        'iters_tail': tail_filter_iters,
        'strel_tail': strel_tail,
        'iters_min': iters_min,
        'strel_min': strel_min,
        'progress_bar': progress_bar,
    }

    if sam2:
        # get the sam2 predictor
        predictor = get_sam2_predictor(sam2_checkpoint)
        # TODO if somehow detect centroid if not DLC keypoints

        # get masks from sam2
        masks, _ = segment_chunk(
            chunk, 
            predictor, 
            sam2_points, 
            clean_params, 
            inference_state=None
            )
        # apply masks to chunk
        chunk = chunk * masks
    

    # Denoise the frames before we do anything else
    filtered_frames = clean_frames(
        chunk,
        prefilter_space=spatial_filter_size,
        prefilter_time=temporal_filter_size,
        iters_tail=tail_filter_iters,
        strel_tail=strel_tail,
        iters_min=iters_min,
        strel_min=strel_min,
        # frame_dtype=frame_dtype,
        progress_bar=progress_bar,
    )

    # now get the centroid and orientation of the mouse
    features, mask = get_frame_features(
        filtered_frames,
        frame_threshold=min_height,
        mask=ll,
        mask_threshold=mask_threshold,
        use_cc=use_cc,
        progress_bar=progress_bar,
    )

    incl = ~np.isnan(features["orientation"])
    features["orientation"][incl] = np.unwrap(features["orientation"][incl] * 2) / 2

    # Crop and rotate the original frames
    cropped_frames = crop_and_rotate_frames(
        chunk, features, crop_size=crop_size, progress_bar=progress_bar
    )

    # Crop and rotate the filtered frames to be returned and later written
    cropped_filtered_frames = crop_and_rotate_frames(
        filtered_frames, features, crop_size=crop_size, progress_bar=progress_bar
    )

    # Compute crop-rotated frame mask
    if use_tracking_model:
        use_parameters = deepcopy(parameters)
        use_parameters["mean"][:, 0] = crop_size[1] // 2
        use_parameters["mean"][:, 1] = crop_size[0] // 2
        mask = em_get_ll(cropped_frames, progress_bar=progress_bar, **use_parameters)
    else:
        mask = crop_and_rotate_frames(
            mask, features, crop_size=crop_size, progress_bar=progress_bar
        )

    # Orient mouse to face east
    if flip_classifier:
        # get frame indices of incorrectly orientation
        flips = get_flips(
            cropped_filtered_frames, flip_classifier, flip_classifier_smoothing
        )
        flip_indices = np.where(flips)

        # apply flips
        cropped_frames[flip_indices] = np.rot90(
            cropped_frames[flip_indices], k=2, axes=(1, 2)
        )
        cropped_filtered_frames[flip_indices] = np.rot90(
            cropped_filtered_frames[flip_indices], k=2, axes=(1, 2)
        )
        mask[flip_indices] = np.rot90(mask[flip_indices], k=2, axes=(1, 2))
        features["orientation"][flips] += np.pi

    else:
        flips = None

    if compute_raw_scalars:
        # Computing scalars from raw data
        scalars = compute_scalars(
            cropped_frames,
            features,
            min_height=min_height,
            max_height=max_height,
            true_depth=true_depth,
        )
    else:
        # Computing scalars from filtered data
        scalars = compute_scalars(
            cropped_filtered_frames,
            features,
            min_height=min_height,
            max_height=max_height,
            true_depth=true_depth,
        )

    # Store all results in a dictionary
    results = {
        "chunk": chunk,
        "depth_frames": cropped_frames,
        "mask_frames": mask,
        "scalars": scalars,
        "flips": flips,
        "parameters": parameters,
    }

    return results
