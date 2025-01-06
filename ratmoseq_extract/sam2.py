from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import cv2
import tempfile
from moseq2_extract.extract.proc import clean_frames
from sam2.build_sam import build_sam2_video_predictor


def load_dlc(csv, bodyparts, frame_range, roi):
    """
    Load DLC data from a CSV file and extract the specified body parts.

    Parameters:
    - csv: str or Path
        Path to the CSV file containing the DLC data.
    - bodyparts: list of str
        List of body parts to extract from the DLC data.
    - frame_range: slice
        Slice object specifying the range of frames to extract.

    Returns:
    - input_point: numpy.ndarray
        2D numpy array with shape (n_points, 2) containing the x and y coordinates of the body parts.
    - input_label: numpy.ndarray
        1D numpy array with shape (n_points,) containing the labels of the body parts.
    """

    dlc = pd.read_csv(csv, index_col=0, header=[1, 2])
    dlc.columns = [f"{c[0]}_{c[1]}" for c in dlc.columns]
    keypoints_df = dlc[[c for c in dlc.columns if '_x' in c or '_y' in c]]
    keypoints_df = keypoints_df[[c for c in keypoints_df.columns if c in bodyparts]]
    keypoints_df = keypoints_df.loc[frame_range]
    input_point = keypoints_df.iloc[0, :].values.reshape(-1, 2)
    # subtract the ROI offset
    input_point -= roi[:2]

    return input_point

def save_frames_to_jpg(array, output_folder, base_filename="frame"):
    """
    Saves each frame in the time axis of a 3D numpy array as a JPG file.

    Parameters:
    - array: numpy.ndarray
        A 3D numpy array with shape (time, width, height).
    - output_folder: str or Path
        Path to the folder where the JPG files will be saved.
    - base_filename: str, optional
        Base name for the saved JPG files. Default is 'frame'.

    Returns:
    None
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    if array.ndim != 3:
        raise ValueError("Input array must be 3D with shape (time, width, height).")

    # Ensure the output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Iterate through the time axis
    for t in range(array.shape[0]):
        frame = array[t]

        # Normalize frame values to 0-255 if not already in that range
        if array.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Create the file path
        file_path = output_folder / f"{t:05d}.jpg"

        # Save the frame as a JPG file
        cv2.imwrite(str(file_path), frame)

    return None

def get_sam2_predictor(sam2_checkpoint):
    """
    Loads a SAM2 predictor from a checkpoint file.

    Parameters:
    - sam2_checkpoint: str or Path
        Path to the SAM2 checkpoint file.
    
    Returns:
    - predictor: sam2_predictor
        SAM2 predictor object.
    """

    assert Path(sam2_checkpoint).exists(), f"Checkpoint file not found: {sam2_checkpoint}"
    assert 'tiny' in sam2_checkpoint, "Only the tiny model is supported for now. Please use a tiny model checkpoint."
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    # make cpu compatible
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    return predictor

def segment_chunk(chunk, predictor, points, clean_params=None, inference_state=None):
    """
    Segment a chunk of video frames using a SAM2 predictor.

    Parameters:
    - chunk: numpy.ndarray
        4D numpy array with shape (time, width, height, channels) containing the video frames.
    - predictor: sam2_predictor
        SAM2 predictor object.
    - points: numpy.ndarray
        2D numpy array with shape (n_points, 2) containing the x and y coordinates of the body parts.
    - clean_params: dict, optional
        Dictionary of parameters to pass to the frame cleaning function.
    - inference_state: dict, optional
        Dictionary containing the inference state of the predictor.

    Returns:
    - masks: numpy.ndarray
        3D numpy array with shape (time, width, height) containing the segmented masks.
    - inference_state: dict
        Dictionary containing the inference state of the predictor
    """

    if inference_state is not None:
        predictor.reset_state(inference_state)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_frames_to_jpg(chunk, tmpdirname)

        inference_state = predictor.init_state(video_path=tmpdirname)

        input_point = points.values.reshape(-1, 2)
        input_label = np.ones(input_point.shape[0])

        _, _, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id='mouse',
            points=input_point,
            labels=input_label,
        )

        # run propagation throughout the video and collect the results in a dict
        masks = []
        for _, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            if clean_params is not None:
                mask = clean_frames(mask, **clean_params)
            masks.append(mask)
        masks = np.array(masks)

    return masks, inference_state
