import numpy as np
import cv2
import subprocess
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
import tarfile
import json
import warnings
import os
import tifffile
import urllib
from typing import Pattern
import re
from cytoolz import keymap
import click
import h5py
from pkg_resources import get_distribution
from glob import glob
from os.path import exists, join, dirname, basename, abspath, splitext
from toolz import valmap
import shutil
import ast

yaml = YAML(typ='safe', pure=True)

def generate_missing_metadata(sess_dir, sess_name):
    """
    Generate metadata.json with default avlues for session that does not already include one.

    Args:
    sess_dir (str): Path to session directory to create metadata.json file in.
    sess_name (str): Session Name to set the metadata SessionName.

    Returns:
    """

    # generate sample metadata json for each session that is missing one
    sample_meta = {
        "SubjectName": "",
        f"SessionName": f"{sess_name}",
        "NidaqChannels": 0,
        "NidaqSamplingRate": 0.0,
        "DepthResolution": [512, 424],
        "ColorDataType": "Byte[]",
        "StartTime": "",
    }

    with open(join(sess_dir, "metadata.json"), "w") as fp:
        json.dump(sample_meta, fp)


def load_timestamps_from_movie(input_file, threads=8, mapping="DEPTH"):
    """
    Run a ffprobe command to extract the timestamps from the .mkv file, and pipes the
    output data to a csv file.

    Args:
    filename (str): path to input file to extract timestamps from.
    threads (int): number of threads to simultaneously read timestamps
    mapping (str): chooses the stream to read from mkv files. (Will default to if video is not an mkv format)

    Returns:
    timestamps (list): list of float values representing timestamps for each frame.
    """

    print("Loading movie timestamps")

    if isinstance(mapping, str):
        mapping_dict = get_stream_names(input_file)
        mapping = mapping_dict.get(mapping, 0)

    command = [
        "ffprobe",
        "-select_streams",
        f"v:{mapping}",
        "-threads",
        str(threads),
        "-show_entries",
        "frame=pkt_pts_time",
        "-v",
        "quiet",
        input_file,
        "-of",
        "csv=p=0",
    ]

    ffprobe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffprobe.communicate()

    if err:
        print("Error:", err)
        return None

    timestamps = [float(t) for t in out.split()]

    if len(timestamps) == 0:
        return None

    return timestamps


def load_metadata(metadata_file):
    """
    Load metadata from session metadata.json file.

    Args:
    metadata_file (str): path to metadata file

    Returns:
    metadata (dict): metadata dictionary of JSON contents
    """

    try:
        if not exists(metadata_file):
            generate_missing_metadata(
                dirname(metadata_file), basename(dirname(metadata_file))
            )

        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    except TypeError:
        # try loading directly
        metadata = json.load(metadata_file)

    return metadata


def load_timestamps(timestamp_file, col=0, alternate=False):
    """
    Read timestamps from space delimited text file for timestamps.

    Args:
    timestamp_file (str): path to timestamp file
    col (int): column in ts file read.
    alternate (boolean): specified if timestamps were saved in a csv file. False means txt file and True means csv file.

    Returns:
    ts (1D array): list of timestamps
    """

    ts = []
    try:
        with open(timestamp_file, "r") as f:
            for line in f:
                cols = line.split()
                ts.append(float(cols[col]))
        ts = np.array(ts)
    except TypeError as e:
        # try iterating directly
        for line in timestamp_file:
            cols = line.split()
            ts.append(float(cols[col]))
        ts = np.array(ts)
    except FileNotFoundError as e:
        ts = None
        warnings.warn(
            "Timestamp file was not found! Make sure the timestamp file exists is named "
            '"depth_ts.txt" or "timestamps.csv".'
        )
        warnings.warn(
            "This could cause issues for large number of dropped frames during the PCA step while "
            "imputing missing data."
        )

    # if timestamps were saved in a csv file
    if alternate:
        ts = ts * 1000

    return ts


def read_yaml(yaml_file):
    """
    Read yaml file into a dictionary

    Args:
    yaml_file (str): path to yaml file

    Returns:
    return_dict (dict): dict of yaml contents
    """

    with open(yaml_file, "r") as f:
        return yaml.load(f)


def dict_to_h5(h5, dic, root="/", annotations=None):
    """
    Save an dict to an h5 file, mounting at root.
    Keys are mapped to group names recursively.

    Args:
    h5 (h5py.File instance): h5py.file object to operate on
    dic (dict): dictionary of data to write
    root (string): group on which to add additional groups and datasets
    annotations (dict): annotation data to add to corresponding h5 datasets. Should contain same keys as dic.

    """

    if not root.endswith("/"):
        root = root + "/"

    if annotations is None:
        annotations = (
            {}
        )  # empty dict is better than None, but dicts shouldn't be default parameters

    for key, item in dic.items():
        dest = root + key
        try:
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                h5[dest] = item
            elif isinstance(item, (tuple, list)):
                h5[dest] = np.asarray(item)
            elif isinstance(item, (int, float)):
                h5[dest] = np.asarray([item])[0]
            elif item is None:
                h5.create_dataset(
                    dest, data=h5py.Empty(dtype=h5py.special_dtype(vlen=str))
                )
            elif isinstance(item, dict):
                dict_to_h5(h5, item, dest)
            else:
                raise ValueError(
                    "Cannot save {} type to key {}".format(type(item), dest)
                )
        except Exception as e:
            print(e)
            if key != "inputs":
                print("h5py could not encode key:", key)

        if key in annotations:
            if annotations[key] is None:
                h5[dest].attrs["description"] = ""
            else:
                h5[dest].attrs["description"] = annotations[key]


def click_param_annot(click_cmd):
    """
    Return a dict that maps option names to help strings from a click.Command instance.

    Args:
    click_cmd (click.Command): command to annotate

    Returns:
    annotations (dict): dictionary of options and their help messages
    """

    annotations = {}
    for p in click_cmd.params:
        if isinstance(p, click.Option):
            annotations[p.human_readable_name] = p.help
    return annotations


def create_extract_h5(
    h5_file,
    acquisition_metadata,
    config_data,
    status_dict,
    scalars_attrs,
    nframes,
    roi,
    bground_im,
    first_frame,
    first_frame_idx,
    last_frame_idx,
    **kwargs,
):
    """
    write acquisition metadata, extraction metadata, computed scalars, timestamps, and original frames/frames_mask to extracted h5.

    Args:
    h5_file (h5py.File object): opened h5 file object to write to.
    acquisition_metadata (dict): Dictionary containing extracted session acquisition metadata.
    config_data (dict): dictionary object containing all required extraction parameters. (auto generated)
    status_dict (dict): dictionary that helps indicate if the session has been extracted fully.
    scalars_attrs (dict): dict of computed scalar attributes and descriptions to save.
    nframes (int): number of frames being recorded
    roi (np.ndarray): Computed 2D ROI Image.
    bground_im (np.ndarray): Computed 2D Background Image.
    first_frame (np.ndarray): Computed 2D First Frame Image.
    timestamps (numpy.array): Array of session timestamps.
    kwargs (dict): additional keyword arguments.

    """

    h5_file.create_dataset("metadata/uuid", data=status_dict["uuid"])

    # Creating scalar dataset
    for scalar in list(scalars_attrs.keys()):
        h5_file.create_dataset(
            f"scalars/{scalar}", (nframes,), "float32", compression="gzip"
        )
        h5_file[f"scalars/{scalar}"].attrs["description"] = scalars_attrs[scalar]

    # Timestamps
    if config_data.get("timestamps") is not None:
        h5_file.create_dataset(
            "timestamps",
            compression="gzip",
            data=config_data["timestamps"][first_frame_idx:last_frame_idx],
        )
        h5_file["timestamps"].attrs["description"] = "Depth video timestamps"

    # Cropped Frames
    h5_file.create_dataset(
        "frames",
        (nframes, config_data["crop_size"][0], config_data["crop_size"][1]),
        config_data["frame_dtype"],
        compression="gzip",
    )
    h5_file["frames"].attrs["description"] = (
        "3D Numpy array of depth frames (nframes x w x h)." + " Depth values are in mm."
    )
    # Frame Masks for EM Tracking
    if config_data["sam2_checkpoint"]:
        h5_file.create_dataset(
            "frames_mask",
            (nframes, config_data["crop_size"][0], config_data["crop_size"][1]),
            "float32",
            compression="gzip",
        )
        h5_file["frames_mask"].attrs[
            "description"
        ] = "Log-likelihood values from the tracking model (nframes x w x h)"
    else:
        h5_file.create_dataset(
            "frames_mask",
            (nframes, config_data["crop_size"][0], config_data["crop_size"][1]),
            "bool",
            compression="gzip",
        )
        h5_file["frames_mask"].attrs[
            "description"
        ] = "Boolean mask, false=not mouse, true=mouse"

    # Flip Classifier
    if config_data["flip_classifier"] is not None:
        h5_file.create_dataset(
            "metadata/extraction/flips", (nframes,), "bool", compression="gzip"
        )
        h5_file["metadata/extraction/flips"].attrs[
            "description"
        ] = "Output from flip classifier, false=no flip, true=flip"

    # True Depth
    h5_file.create_dataset(
        "metadata/extraction/true_depth", data=config_data["true_depth"]
    )
    h5_file["metadata/extraction/true_depth"].attrs[
        "description"
    ] = "Detected true depth of arena floor in mm"

    # ROI
    h5_file.create_dataset("metadata/extraction/roi", data=roi, compression="gzip")
    h5_file["metadata/extraction/roi"].attrs["description"] = "ROI mask"

    # First Frame
    h5_file.create_dataset(
        "metadata/extraction/first_frame", data=first_frame[0], compression="gzip"
    )
    h5_file["metadata/extraction/first_frame"].attrs[
        "description"
    ] = "First frame of depth dataset"

    # First Frame index
    h5_file.create_dataset(
        "metadata/extraction/first_frame_idx",
        data=[first_frame_idx],
        compression="gzip",
    )
    h5_file["metadata/extraction/first_frame_idx"].attrs[
        "description"
    ] = "First frame index of this dataset"

    # Last Frame index
    h5_file.create_dataset(
        "metadata/extraction/last_frame_idx", data=[last_frame_idx], compression="gzip"
    )
    h5_file["metadata/extraction/last_frame_idx"].attrs[
        "description"
    ] = "Last frame index of this dataset"

    # Background
    h5_file.create_dataset(
        "metadata/extraction/background", data=bground_im, compression="gzip"
    )
    h5_file["metadata/extraction/background"].attrs[
        "description"
    ] = "Computed background image"

    # Extract Version
    extract_version = np.string_(get_distribution("ratmoseq-extract").version)
    h5_file.create_dataset("metadata/extraction/extract_version", data=extract_version)
    h5_file["metadata/extraction/extract_version"].attrs[
        "description"
    ] = "Version of moseq2-extract"

    # Extraction Parameters
    from ratmoseq_extract.cli import extract

    dict_to_h5(
        h5_file,
        status_dict["parameters"],
        "metadata/extraction/parameters",
        click_param_annot(extract),
    )

    # Acquisition Metadata
    for key, value in acquisition_metadata.items():
        if type(value) is list and len(value) > 0 and type(value[0]) is str:
            value = [n.encode("utf8") for n in value]

        if value is not None:
            h5_file.create_dataset(f"metadata/acquisition/{key}", data=value)
        else:
            h5_file.create_dataset(f"metadata/acquisition/{key}", dtype="f")


def write_extracted_chunk_to_h5(
    h5_file, results, config_data, scalars, frame_range, offset
):
    """

    Write extracted frames, frame masks, and scalars to an open h5 file.

    Args:
    h5_file (H5py.File): open results_00 h5 file to save data in.
    results (dict): extraction results dict.
    config_data (dict): dictionary containing extraction parameters (autogenerated)
    scalars (list): list of keys to scalar attribute values
    frame_range (range object): current chunk frame range
    offset (int): frame offset

    Returns:
    """

    # Writing computed scalars to h5 file
    for scalar in scalars:
        h5_file[f"scalars/{scalar}"][frame_range] = results["scalars"][scalar][offset:]

    # Writing frames and mask to h5
    h5_file["frames"][frame_range] = results["depth_frames"][offset:]
    h5_file["frames_mask"][frame_range] = results["mask_frames"][offset:]

    # Writing flip classifier results to h5
    if config_data["flip_classifier"]:
        h5_file["metadata/extraction/flips"][frame_range] = results["flips"][offset:]


def make_output_movie(results, config_data, offset=0):
    """
    Create an array for output movie with filtered video and cropped mouse on the top left

    Args:
    results (dict): dict of extracted depth frames, and original raw chunk to create an output movie.
    config_data (dict): dict of extraction parameters containing the crop sizes used in the extraction.
    offset (int): current offset being used, automatically set if chunk_overlap > 0

    Returns:
    output_movie (numpy.ndarray): output movie to write to mp4 file.
    """

    # Create empty array for output movie with filtered video and cropped mouse on the top left
    nframes, rows, cols = results["chunk"][offset:].shape
    output_movie = np.zeros(
        (
            nframes,
            rows + config_data["crop_size"][0],
            cols + config_data["crop_size"][1],
        ),
        "uint16",
    )

    # Populating array with filtered and cropped videos
    output_movie[:, : config_data["crop_size"][0], : config_data["crop_size"][1]] = (
        results["depth_frames"][offset:]
    )
    output_movie[:, config_data["crop_size"][0] :, config_data["crop_size"][1] :] = (
        results["chunk"][offset:]
    )

    # normalize from 0-255 for writing to mp4
    output_movie = (output_movie / output_movie.max() * 255).astype("uint8")

    return output_movie


def handle_extract_metadata(input_file, dirname):
    """
    Extract metadata and timestamp in the extraction.

    Args:
    input_file (str): path to input file to extract
    dirname (str): path to directory where extraction files reside.

    Returns:
    acquisition_metadata (dict): key-value pairs of JSON contents
    timestamps (1D array): list of loaded timestamps
    tar (bool): indicator for whether the file is compressed.
    """

    tar = None
    tar_members = None
    alternate_correct = False
    from_depth_file = False

    # Handle TAR files
    if input_file.endswith((".tar.gz", ".tgz")):
        print(f"Scanning tarball {input_file} (this will take a minute)")
        # compute NEW psuedo-dirname now, `input_file` gets overwritten below with test_vid.dat tarinfo...
        dirname = join(
            dirname, basename(input_file).replace(".tar.gz", "").replace(".tgz", "")
        )

        tar = tarfile.open(input_file, "r:gz")
        tar_members = tar.getmembers()
        tar_names = [_.name for _ in tar_members]

    if tar is not None:
        # Handling tar paths
        metadata_path = tar.extractfile(tar_members[tar_names.index("metadata.json")])
        if "depth_ts.txt" in tar_names:
            timestamp_path = tar.extractfile(
                tar_members[tar_names.index("depth_ts.txt")]
            )
        elif "timestamps.csv" in tar_names:
            timestamp_path = tar.extractfile(
                tar_members[tar_names.index("timestamps.csv")]
            )
            alternate_correct = True
    else:
        # Handling non-compressed session paths
        metadata_path = join(dirname, "metadata.json")
        timestamp_path = join(dirname, "depth_ts.txt")
        alternate_timestamp_path = join(dirname, "timestamps.csv")
        # Checks for alternative timestamp file if original .txt extension does not exist
        if not exists(timestamp_path) and exists(alternate_timestamp_path):
            timestamp_path = alternate_timestamp_path
            alternate_correct = True
        elif not (
            exists(timestamp_path) or exists(alternate_timestamp_path)
        ) and input_file.endswith(".mkv"):
            from_depth_file = True

    acquisition_metadata = load_metadata(metadata_path)
    if not from_depth_file:
        timestamps = load_timestamps(timestamp_path, col=0, alternate=alternate_correct)
    else:
        timestamps = load_timestamps_from_movie(input_file)

    return acquisition_metadata, timestamps, tar


def get_raw_info(filename, bit_depth=16, frame_size=(512, 424)):
    """
    Get info from a raw data file with specified frame dimensions and bit depth.

    Args:
    filename (str): name of raw data file
    bit_depth (int): bits per pixel (default: 16)
    frame_dims (tuple): wxh or hxw of each frame

    Returns:
    file_info (dict): dictionary containing depth file metadata
    """

    bytes_per_frame = (frame_size[0] * frame_size[1] * bit_depth) / 8

    if type(filename) is not tarfile.TarFile:
        file_info = {
            "bytes": os.stat(filename).st_size,
            "nframes": int(os.stat(filename).st_size / bytes_per_frame),
            "dims": frame_size,
            "bytes_per_frame": bytes_per_frame,
        }
    else:
        tar_members = filename.getmembers()
        tar_names = [_.name for _ in tar_members]
        input_file = tar_members[tar_names.index("depth.dat")]
        file_info = {
            "bytes": input_file.size,
            "nframes": int(input_file.size / bytes_per_frame),
            "dims": frame_size,
            "bytes_per_frame": bytes_per_frame,
        }
    return file_info


def get_movie_info(
    filename, frame_size=(512, 424), bit_depth=16, mapping="DEPTH", threads=8, **kwargs
):
    """
    Return dict of movie metadata.

    Args:
    filename (str): path to video file
    frame_dims (tuple): video dimensions
    bit_depth (int): integer indicating data type encoding
    mapping (str): the stream to read from mkv files
    threads (int): number of threads to simultaneously read timestamps stored within the raw data file.

    Returns:
    metadata (dict): dictionary containing video file metadata
    """

    try:
        if type(filename) is tarfile.TarFile:
            metadata = get_raw_info(
                filename, frame_size=frame_size, bit_depth=bit_depth
            )
        elif filename.lower().endswith(".dat"):
            metadata = get_raw_info(
                filename, frame_size=frame_size, bit_depth=bit_depth
            )
        elif filename.lower().endswith((".avi", ".mkv")):
            metadata = get_video_info(
                filename, mapping=mapping, threads=threads, **kwargs
            )
    except AttributeError as e:
        print("Error reading movie metadata:", e)
        metadata = {}

    return metadata


def get_frame_range_indices(trim_beginning, trim_ending, nframes):
    """
    Compute the total number of frames to be extracted, and find the start and end indices.

    Args:
    trim_beginning (int): number of frames to remove from beginning of recording
    trim_ending (int): number of frames to remove from ending of recording
    nframes (int): total number of requested frames to extract

    Returns:
    nframes (int): total number of frames to extract
    first_frame_idx (int): index of the frame to begin extraction from
    last_frame_idx (int): index of the last frame in the extraction
    """
    assert all(
        (trim_ending >= 0, trim_beginning >= 0)
    ), "frame_trim arguments must be greater than or equal to 0!"

    first_frame_idx = 0
    if trim_beginning > 0 and trim_beginning < nframes:
        first_frame_idx = trim_beginning

    last_frame_idx = nframes
    if first_frame_idx < (nframes - trim_ending) and trim_ending > 0:
        last_frame_idx = nframes - trim_ending

    total_frames = last_frame_idx - first_frame_idx

    return total_frames, first_frame_idx, last_frame_idx


def scalar_attributes():
    """
    Gets scalar attributes dict with names paired with descriptions.

    Returns:
    attributes (dict): a dictionary of metadata keys and descriptions.
    """

    attributes = {
        "centroid_x_px": "X centroid (pixels)",
        "centroid_y_px": "Y centroid (pixels)",
        "velocity_2d_px": "2D velocity (pixels / frame), note that missing frames are not accounted for",
        "velocity_3d_px": "3D velocity (pixels / frame), note that missing frames are not accounted for, also height is in mm, not pixels for calculation",
        "width_px": "Mouse width (pixels)",
        "length_px": "Mouse length (pixels)",
        "area_px": "Mouse area (pixels)",
        "centroid_x_mm": "X centroid (mm)",
        "centroid_y_mm": "Y centroid (mm)",
        "velocity_2d_mm": "2D velocity (mm / frame), note that missing frames are not accounted for",
        "velocity_3d_mm": "3D velocity (mm / frame), note that missing frames are not accounted for",
        "width_mm": "Mouse width (mm)",
        "length_mm": "Mouse length (mm)",
        "area_mm": "Mouse area (mm)",
        "height_ave_mm": "Mouse average height (mm)",
        "angle": "Angle (radians, unwrapped)",
        "velocity_theta": "Angular component of velocity (arctan(vel_x, vel_y))",
    }

    return attributes


def gen_batch_sequence(nframes, chunk_size, overlap, offset=0):
    """
    Generates batches used to chunk videos prior to extraction.

    Args:
    nframes (int): total number of frames
    chunk_size (int): the number of desired chunk size
    overlap (int): number of overlapping frames
    offset (int): frame offset

    Returns:
    out (list): the list of batches
    """

    seq = range(offset, nframes)
    out = []
    for i in range(0, len(seq) - overlap, chunk_size - overlap):
        out.append(seq[i : i + chunk_size])
    return out


def read_yaml(yaml_file):
    """
    Read yaml file into a dictionary

    Args:
    yaml_file (str): path to yaml file

    Returns:
    return_dict (dict): dict of yaml contents
    """

    with open(yaml_file, "r") as f:
        return yaml.load(f)


def write_image(
    filename, image, scale=True, scale_factor=None, frame_dtype="uint16"
):
    """
    Save image data.

    Args:
    filename (str): path to output file
    image (numpy.ndarray): the (unscaled) 2-D image to save
    scale (bool): flag to scale the image between the bounds of `dtype`
    scale_factor (int): factor by which to scale image
    frame_dtype (str): array data type
    compress (int): image compression level

    """

    file = filename

    metadata = {}

    if scale:
        max_int = np.iinfo(frame_dtype).max

        if not scale_factor:
            # scale image to `dtype`'s full range
            scale_factor = int(
                max_int / (np.nanmax(image) + 1e-25)
            )  # adding very small value to avoid divide by 0
            image = image * scale_factor
        elif isinstance(scale_factor, tuple):
            image = np.float32(image)
            image = (image - scale_factor[0]) / (scale_factor[1] - scale_factor[0])
            image = np.clip(image, 0, 1) * max_int

        metadata = {"scale_factor": str(scale_factor)}

    directory = dirname(file)
    if not exists(directory):
        os.makedirs(directory)

    tifffile.imsave(
        file, image.astype(frame_dtype), metadata=metadata
    )


def write_frames_preview(
    filename,
    frames=np.empty((0,)),
    threads=6,
    fps=30,
    pixel_format="rgb24",
    codec="h264",
    slices=24,
    slicecrc=1,
    frame_size=None,
    depth_min=0,
    depth_max=80,
    get_cmd=False,
    cmap="jet",
    pipe=None,
    close_pipe=True,
    frame_range=None,
    progress_bar=False,
):
    """
    Simple command to pipe frames to an ffv1 file. Writes out a false-colored mp4 video.

    Args:
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format video color scheme
    codec (str): ffmpeg encoding-writer method to use
    slices (int): number of frame slices to write at a time.
    slicecrc (int): check integrity of slices
    frame_size (tuple): shape/dimensions of image.
    depth_min (int): minimum mouse depth from floor in (mm)
    depth_max (int): maximum mouse depth from floor in (mm)
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing)
    cmap (str): color map to use.
    pipe (subProcess.Pipe): pipe to currently open video file.
    close_pipe (bool): indicates to close the open pipe to video when done writing.
    frame_range (range()): frame indices to write on video
    progress_bar (bool): If True, displays a TQDM progress bar for the video writing progress.

    Returns:
    pipe (subProcess.Pipe): indicates whether video writing is complete.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    txt_pos = (5, frames.shape[-1] - 40)

    if not np.mod(frames.shape[1], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 1), (0, 0)), "constant", constant_values=0)

    if not np.mod(frames.shape[2], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=0)

    if not frame_size and type(frames) is np.ndarray:
        frame_size = "{0:d}x{1:d}".format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = "{0:d}x{1:d}".format(frames[0], frames[1])

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "fatal",
        "-threads",
        str(threads),
        "-framerate",
        str(fps),
        "-f",
        "rawvideo",
        "-s",
        frame_size,
        "-pix_fmt",
        pixel_format,
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-slices",
        str(slices),
        "-slicecrc",
        str(slicecrc),
        "-r",
        str(fps),
        filename,
    ]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # scale frames to appropriate depth ranges
    use_cmap = plt.get_cmap(cmap)
    for i in tqdm(
        range(frames.shape[0]),
        disable=not progress_bar,
        desc=f"Writing frames to {filename}",
    ):
        disp_img = frames[i, :].copy().astype("float32")
        disp_img = (disp_img - depth_min) / (depth_max - depth_min)
        disp_img[disp_img < 0] = 0
        disp_img[disp_img > 1] = 1
        disp_img = np.delete(use_cmap(disp_img), 3, 2) * 255
        if frame_range is not None:
            try:
                cv2.putText(
                    disp_img,
                    str(frame_range[i]),
                    txt_pos,
                    font,
                    1,
                    white,
                    2,
                    cv2.LINE_AA,
                )
            except (IndexError, ValueError):
                # len(frame_range) M < len(frames) or
                # txt_pos is outside of the frame dimensions
                print("Could not overlay frame number on preview on video.")

        pipe.stdin.write(disp_img.astype("uint8").tostring())

    if close_pipe:
        pipe.communicate()
        return None
    else:
        return pipe


def read_frames(
    filename,
    frames=range(
        0,
    ),
    threads=6,
    fps=30,
    frames_is_timestamp=False,
    pixel_format="gray16le",
    movie_dtype="uint16",
    frame_size=None,
    slices=24,
    slicecrc=1,
    mapping="DEPTH",
    get_cmd=False,
    finfo=None,
    **kwargs,
):
    """
    Read in frames from the .mp4/.avi file using a pipe from ffmpeg.

    Args:
    filename (str): filename to get frames from
    frames (list or numpy.ndarray): list of frames to grab
    threads (int): number of threads to use for decode
    fps (int): frame rate of camera
    frames_is_timestamp (bool): if False, indicates timestamps represent kinect v2 absolute machine timestamps,
    pixel_format (str): ffmpeg pixel format of data
    movie_dtype (str): An indicator for numpy to store the piped ffmpeg-read video in memory for processing.
    frame_size (str): wxh frame size in pixels
    slices (int): number of slices to use for decode
    slicecrc (int): check integrity of slices
    mapping (str): the stream to read from mkv files.
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing).
    finfo (dict): dictionary containing video file metadata

    Returns:
    video (numpy.ndarray):  frames x rows x columns
    """

    if finfo is None:
        finfo = get_video_info(filename, threads=threads, **kwargs)

    if frames is None or len(frames) == 0:
        frames = np.arange(finfo["nframes"], dtype="int64")

    if not frame_size:
        frame_size = finfo["dims"]

    # Compute starting time point to retrieve frames from
    if frames_is_timestamp:
        start_time = str(datetime.timedelta(seconds=frames[0]))
    else:
        start_time = str(datetime.timedelta(seconds=frames[0] / fps))

    command = [
        "ffmpeg",
        "-loglevel",
        "fatal",
        "-ss",
        start_time,
        "-i",
        filename,
        "-vframes",
        str(len(frames)),
        "-f",
        "image2pipe",
        "-s",
        "{:d}x{:d}".format(frame_size[0], frame_size[1]),
        "-pix_fmt",
        pixel_format,
        "-threads",
        str(threads),
        "-slices",
        str(slices),
        "-slicecrc",
        str(slicecrc),
        "-vcodec",
        "rawvideo",
    ]

    if isinstance(mapping, str):
        mapping_dict = get_stream_names(filename)
        mapping = mapping_dict.get(mapping, 0)

    if filename.endswith((".mkv", ".avi")):
        command += ["-map", f"0:{mapping}"]
        command += ["-vsync", "0"]

    command += ["-"]

    if get_cmd:
        return command

    pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = pipe.communicate()

    if err:
        print("Error:", err)
        return None

    video = np.frombuffer(out, dtype=movie_dtype).reshape(
        (len(frames), frame_size[1], frame_size[0])
    )

    return video.astype("uint16")


def get_stream_names(filename, stream_tag="title"):
    """
    Run an FFProbe command to determine whether an input video file contains multiple streams, and
    returns a stream_name to paired int values to extract the desired stream.

    Args:
    filename (str): path to video file to get streams from.
    stream_tag (str): value of the stream tags for ffprobe command to return

    Returns:
    out (dict): Dictionary of string to int pairs for the included streams in the mkv file.
    Dict will be used to choose the correct mapping number to choose which stream to read in read_frames().
    """

    command = [
        "ffprobe",
        "-v",
        "fatal",
        "-show_entries",
        "stream_tags={}".format(stream_tag),
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        filename,
    ]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()

    if err or len(out) == 0:
        return {"DEPTH": 0}

    out = out.decode("utf-8").rstrip("\n").split("\n")

    return {o: i for i, o in enumerate(out)}


def get_video_info(filename, mapping="DEPTH", threads=8, count_frames=False, **kwargs):
    """
    Get file metadata from videos.

    Args:
    filename (str): name of file to read video metadata from.
    mapping (str): chooses the stream to read from files.
    threads (int): number of threads to simultanoues run the ffprobe command
    count_frames (bool): indicates whether to count the frames individually.

    Returns:
    out_dict (dict): dictionary containing video file metadata
    """

    mapping_dict = get_stream_names(filename)
    if isinstance(mapping, str):
        mapping = mapping_dict.get(mapping, 0)

    stream_str = "stream=width,height,r_frame_rate,"
    if count_frames:
        stream_str += "nb_read_frames"
    else:
        stream_str += "nb_frames"

    command = [
        "ffprobe",
        "-v",
        "fatal",
        "-select_streams",
        f"v:{mapping}",
        "-show_entries",
        stream_str,
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        "-threads",
        str(threads),
        filename,
        "-sexagesimal",
    ]

    if count_frames:
        command += ["-count_frames"]

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()

    if err:
        print(err)

    out = out.decode().split("\n")
    out_dict = {
        "file": filename,
        "dims": (int(float(out[0])), int(float(out[1]))),
        "fps": float(out[2].split("/")[0]) / float(out[2].split("/")[1]),
    }

    try:
        out_dict["nframes"] = int(out[3])
    except ValueError:
        out_dict["nframes"] = None

    return out_dict


def check_completion_status(status_filename):
    """
    Read a results_00.yaml (status file) and checks whether the session has been
    fully extracted.

    Args:
    status_filename (str): path to results_00.yaml

    Returns:
    complete (bool): If True, data has been extracted to completion.
    """

    if exists(status_filename):
        return read_yaml(status_filename)["complete"]
    return False


def recursive_find_unextracted_dirs(
    root_dir=os.getcwd(),
    session_pattern=r"session_\d+\.(?:tgz|tar\.gz)",
    extension=".dat",
    yaml_path="proc/results_00.yaml",
    metadata_path="metadata.json",
    skip_checks=False,
):
    """
    Recursively find unextracted (or incompletely extracted) directories

    Args:
    root_dir (str): path to base directory to start recursive search for unextracted folders.
    session_pattern (str): folder name pattern to search for
    extension (str): file extension to search for
    yaml_path (str): path to respective extracted metadata
    metadata_path (str): path to relative metadata.json files
    skip_checks (bool): indicates whether to check if the files exist at the given relative paths

    Returns:
    proc_dirs (1d-list): list of paths to each unextracted session's proc/ directory
    """

    session_archive_pattern = re.compile(session_pattern)

    proc_dirs = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension) and not file.startswith(
                "ir"
            ):  # test for uncompressed session
                status_file = join(root, yaml_path)
                metadata_file = join(root, metadata_path)
            elif session_archive_pattern.fullmatch(file):  # test for compressed session
                session_name = basename(file).replace(".tar.gz", "").replace(".tgz", "")
                status_file = join(root, session_name, yaml_path)
                metadata_file = join(root, "{}.json".format(session_name))
            else:
                continue  # skip this current file as it does not look like session data

            # perform checks, append depth file to list if extraction is missing or incomplete
            if skip_checks or (
                not check_completion_status(status_file) and exists(metadata_file)
            ):
                proc_dirs.append(join(root, file))

    return proc_dirs


def recursive_find_h5s(root_dir=os.getcwd(), ext=".h5", yaml_string="{}.yaml"):
    """
    Recursively find h5 files, along with yaml files with the same basename

    Args:
    root_dir (str): path to base directory to begin recursive search in.
    ext (str): extension to search for
    yaml_string (str): string for filename formatting when saving data

    Returns:
    h5s (list): list of found h5 files
    dicts (list): list of found metadata files
    yamls (list): list of found yaml files
    """
    if not ext.startswith("."):
        ext = "." + ext

    def has_frames(f):
        try:
            with h5py.File(f, "r") as h5f:
                return "frames" in h5f
        except OSError:
            warnings.warn(f"Error reading {f}, skipping...")
            return False

    h5s = glob(join(abspath(root_dir), "**", f"*{ext}"), recursive=True)
    h5s = filter(lambda f: exists(yaml_string.format(f.replace(ext, ""))), h5s)
    h5s = list(filter(has_frames, h5s))
    yamls = list(map(lambda f: yaml_string.format(f.replace(ext, "")), h5s))
    dicts = list(map(read_yaml, yamls))

    return h5s, dicts, yamls


def clean_dict(dct):
    """
    Standardize types of dict value.

    Args:
    dct (dict): dict object with mixed type value objects.

    Returns:
    out (dict): dict object with list value objects.
    """

    def clean_entry(e):
        if isinstance(e, dict):
            out = clean_dict(e)
        elif isinstance(e, np.ndarray):
            out = e.tolist()
        elif isinstance(e, np.generic):
            out = np.asscalar(e)
        else:
            out = e
        return out

    return valmap(clean_entry, dct)


def _load_h5_to_dict(file: h5py.File, path) -> dict:
    """
    Loads h5 contents to dictionary object.

    Args:
    h5file (h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file

    Returns:
    ans (dict): a dict with h5 file contents with the same path structure
    """

    ans = {}
    for key, item in file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _load_h5_to_dict(file, "/".join([path, key]))
    return ans


def h5_to_dict(h5file, path) -> dict:
    """
    Load h5 contents to dictionary object.

    Args:
    h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file

    Returns:
    out (dict): a dict with h5 file contents with the same path structure
    """

    if isinstance(h5file, str):
        with h5py.File(h5file, "r") as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, h5py.File):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise Exception("file input not understood - need h5 file path or file object")
    return out


def copy_h5_metadata_to_yaml(input_dir, h5_metadata_path):
    """
    Copy user specified metadata from h5path to a yaml file.

    Args:
    input_dir (str): path to directory containing h5 files
    h5_metadata_path (str): path within h5 to desired metadata to copy to yaml.

    """

    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    to_load = [
        (tmp, yml, file)
        for tmp, yml, file in zip(dicts, yamls, h5s)
        if tmp["complete"] and not tmp["skip"]
    ]

    # load in all of the h5 files, grab the extraction metadata, reformat to make nice 'n pretty
    # then stage the copy

    for tup in tqdm(to_load, desc="Copying data to yamls"):
        with h5py.File(tup[2], "r") as f:
            tmp = clean_dict(h5_to_dict(f, h5_metadata_path))
            tup[0]["metadata"] = dict(tmp)

        new_file = f"{basename(tup[1])}_update.yaml"
        with open(new_file, "w+") as f:
            yaml.safe_dump(tup[0], f)

        if new_file != tup[1]:
            shutil.move(new_file, tup[1])


def build_index_dict(files_to_use):
    """
    Create a dictionary for the index file from a list of files and respective metadatas.

    Args:
    files_to_use (list): list of paths to extracted h5 files.

    Returns:
    output_dict (dict): index-file dictionary containing all aggregated extractions.
    """

    output_dict = {"files": [], "pca_path": ""}

    index_uuids = []
    for i, file_tup in enumerate(files_to_use):
        if file_tup[2]["uuid"] not in index_uuids:
            tmp = {
                "path": (file_tup[0], file_tup[1]),
                "uuid": file_tup[2]["uuid"],
                "group": "default",
                "metadata": {
                    "SessionName": f"default_{i}",
                    "SubjectName": f"default_{i}",
                },  # fallback metadata
            }

            # handling metadata sub-dictionary values
            if "metadata" in file_tup[2]:
                tmp["metadata"].update(file_tup[2]["metadata"])
            else:
                warnings.warn(
                    f"Could not locate metadata for {file_tup[0]}! File will be listed with minimal default metadata."
                )

            index_uuids.append(file_tup[2]["uuid"])
            # appending file with default information
            output_dict["files"].append(tmp)

    return output_dict


def filter_warnings(func):
    """
    Applies warnings.simplefilter() to ignore warnings when
     running the main gui functionaity in a Jupyter Notebook.
     The function will filter out: yaml.error.UnsafeLoaderWarning, FutureWarning and UserWarning.

    Args:
    func (function): function to silence enclosed warnings.

    Returns:
    apply_warning_filters (func): Returns passed function after warnings filtering is completed.
    """

    def apply_warning_filters(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", yaml.error.UnsafeLoaderWarning)
            warnings.simplefilter(action="ignore", category=FutureWarning)
            warnings.simplefilter(action="ignore", category=UserWarning)
            return func(*args, **kwargs)

    return apply_warning_filters


@filter_warnings
def generate_index(input_dir, output_file):
    """
    Generate index file containing a summary of all extracted sessions.

    Args:
    input_dir (str): directory to search for extracted sessions.
    output_file (str): preferred name of the index file.

    Returns:
    output_file (str): path to index file (moseq2-index.yaml).
    """

    # gather the h5s and the pca scores file
    # uuids should match keys in the scores file
    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    file_with_uuids = [
        (abspath(h5), abspath(yml), meta) for h5, yml, meta in zip(h5s, yamls, dicts)
    ]

    # Ensuring all retrieved extracted session h5s have the appropriate metadata
    # included in their results_00.h5 file
    for file in file_with_uuids:
        try:
            if "metadata" not in file[2]:
                copy_h5_metadata_to_yaml(input_dir, file[0])
        except:
            warnings.warn(
                f"Metadata for session {file[0]} not found. \
            File may be listed with minimal/defaulted metadata in index file."
            )

    print(f"Number of sessions included in index file: {len(file_with_uuids)}")

    # Create index file in dict form
    output_dict = build_index_dict(file_with_uuids)

    # write out index yaml
    with open(output_file, "w") as f:
        yaml.safe_dump(output_dict, f)

    return output_file


def camel_to_snake(s):
    """
    Convert CamelCase to snake_case

    Args:
    s (str): CamelCase string to convert to snake_case.

    Returns:
    (str): string in snake_case
    """
    _underscorer1: Pattern[str] = re.compile(r"(.)([A-Z][a-z]+)")
    _underscorer2 = re.compile("([a-z0-9])([A-Z])")

    subbed = _underscorer1.sub(r"\1_\2", s)
    return _underscorer2.sub(r"\1_\2", subbed).lower()


def load_extraction_meta_from_h5s(to_load, snake_case=True):
    """
    Load extraction metadata from h5 files.

    Args:
    to_load (list): list of paths to h5 files.
    snake_case (bool): whether to save the files using snake_case

    Returns:
    loaded (list): list of loaded h5 dicts.
    """

    loaded = []
    for _dict, _h5f in tqdm(to_load, desc="Scanning data"):
        try:
            # v0.1.3 introduced a change - acq. metadata now here
            tmp = h5_to_dict(_h5f, "/metadata/acquisition")
        except KeyError:
            # if it doesn't exist it's likely from an older moseq version. Try loading it here
            try:
                tmp = h5_to_dict(_h5f, "/metadata/extraction")
            except KeyError:
                # if all else fails, abandon all hope
                tmp = {}

        # note that everything going into here must be a string (no bytes!)
        tmp = {k: str(v) for k, v in tmp.items()}
        if snake_case:
            tmp = keymap(camel_to_snake, tmp)

        # Specific use case block: Behavior reinforcement experiments
        feedback_file = join(dirname(_h5f), "..", "feedback_ts.txt")
        if exists(feedback_file):
            timestamps = map(int, load_timestamps(feedback_file, 0))
            feedback_status = map(int, load_timestamps(feedback_file, 1))
            _dict["feedback_timestamps"] = list(zip(timestamps, feedback_status))

        _dict["extraction_metadata"] = tmp
        loaded += [(_dict, _h5f)]

    return loaded


def load_textdata(data_file, dtype=np.float32):
    """
    Loads timestamp from txt/csv file.

    Args:
    data_file (str): path to timestamp file
    dtype (dtype): data type of timestamps

    Returns:
    data (np.ndarray): timestamp data
    timestamps (numpy.array): the array for the timestamps
    """

    data = []
    timestamps = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            tmp = line.split(" ", 1)
            # appending timestamp value
            timestamps.append(int(float(tmp[0])))

            # append data indicator value
            clean_data = np.fromstring(
                tmp[1].replace(" ", "").strip(), sep=",", dtype=dtype
            )
            data.append(clean_data)

    data = np.stack(data, axis=0).squeeze()
    timestamps = np.array(timestamps, dtype=np.int)

    return data, timestamps


def time_str_for_filename(time_str: str) -> str:
    """
    Process the timestamp to be used in the filename.

    Args:
    time_str (str): time str to format

    Returns:
    out (str): formatted timestamp str
    """

    out = time_str.split(".")[0]
    out = out.replace(":", "-").replace("T", "_")
    return out


def clean_file_str(file_str: str, replace_with: str = "-") -> str:
    """
    Removes invalid characters for a file name from a string.

    Args:
    file_str (str): filename substring to replace
    replace_with (str): value to replace str with

    Returns:
    out (str): cleaned file string
    """

    out = re.sub(r'[ <>:"/\\|?*\']', replace_with, file_str)
    # find any occurrences of `replace_with`, i.e. (--)
    return re.sub(replace_with * 2, replace_with, out)


def build_path(keys: dict, format_string: str, snake_case=True) -> str:
    """
    Produce a new file name using keys collected from extraction h5 files.

    Args:
    keys (dict): dictionary specifying which keys used to produce the new file name
    format_string (str): the string to reformat using the `keys` dictionary i.e. '{subject_name}_{session_name}'.
    snake_case (bool): flag to save the files with snake_case

    Returns:
    out (str): a newly formatted filename useable with any operating system
    """

    if "start_time" in keys:
        # process the time value
        keys["start_time"] = time_str_for_filename(keys["start_time"])

    if snake_case:
        keys = valmap(camel_to_snake, keys)

    return clean_file_str(format_string.format(**keys))


def build_manifest(loaded, format, snake_case=True):
    """
    Build a manifest file used to contain extraction result metadata from h5 and yaml files.

    Args:
    loaded (list of dicts): list of dicts containing loaded h5 data.
    format (str): filename format indicating the new name for the metadata files in the aggregate_results dir.
    snake_case (bool): whether to save the files using snake_case

    Returns:
    manifest (dict): dictionary of extraction metadata.
    """

    manifest = {}
    fallback = "session_{:03d}"
    fallback_count = 0

    # Additional metadata for certain use cases
    additional_meta = []

    # Behavior reinforcement metadata
    additional_meta.append(
        {
            "filename": "feedback_ts.txt",
            "var_name": "realtime_feedback",
            "dtype": np.bool,
        }
    )

    # Pre-trained model real-time syllable classification results
    additional_meta.append(
        {
            "filename": "predictions.txt",
            "var_name": "realtime_predictions",
            "dtype": np.int,
        }
    )

    # Real-Time Recorded/Computed PC Scores
    additional_meta.append(
        {
            "filename": "pc_scores.txt",
            "var_name": "realtime_pc_scores",
            "dtype": np.float32,
        }
    )

    for _dict, _h5f in loaded:
        print_format = f"{format}_{splitext(basename(_h5f))[0]}"
        if not _dict["extraction_metadata"]:
            copy_path = fallback.format(fallback_count)
            fallback_count += 1
        else:
            try:
                copy_path = build_path(
                    _dict["extraction_metadata"], print_format, snake_case=snake_case
                )
            except:
                copy_path = fallback.format(fallback_count)
                fallback_count += 1
                pass

        # add a bonus dictionary here to be copied to h5 file itself
        manifest[_h5f] = {
            "copy_path": copy_path,
            "yaml_dict": _dict,
            "additional_metadata": {},
        }
        for meta in additional_meta:
            filename = join(dirname(_h5f), "..", meta["filename"])
            if exists(filename):
                try:
                    data, timestamps = load_textdata(filename, dtype=meta["dtype"])
                    manifest[_h5f]["additional_metadata"][meta["var_name"]] = {
                        "data": data,
                        "timestamps": timestamps,
                    }
                except:
                    warnings.warn(
                        "WARNING: Did not load timestamps! This may cause issues if total dropped frames > 2% of the session."
                    )

    return manifest


def copy_manifest_results(manifest, output_dir):
    """
    Copy all consolidated manifest results to their respective output files.

    Args:
    manifest (dict): manifest dictionary containing all extraction h5 metadata to save
    output_dir (str): path to directory where extraction results will be aggregated.

    """

    if not exists(output_dir):
        os.makedirs(output_dir)

    # now the key is the source h5 file and the value is the path to copy to
    for k, v in tqdm(manifest.items(), desc="Copying files"):

        if exists(join(output_dir, f'{v["copy_path"]}.h5')):
            continue

        in_basename = splitext(basename(k))[0]
        in_dirname = dirname(k)

        h5_path = k
        mp4_path = join(in_dirname, f"{in_basename}.mp4")

        if exists(h5_path):
            new_h5_path = join(output_dir, f'{v["copy_path"]}.h5')
            shutil.copyfile(h5_path, new_h5_path)

        # if we have additional_meta then crack open the h5py and write to a safe place
        if len(v["additional_metadata"]) > 0:
            for k2, v2 in v["additional_metadata"].items():
                new_key = f"/metadata/misc/{k2}"
                with h5py.File(new_h5_path, "a") as f:
                    f.create_dataset(f"{new_key}/data", data=v2["data"])
                    f.create_dataset(f"{new_key}/timestamps", data=v2["timestamps"])

        if exists(mp4_path):
            shutil.copyfile(mp4_path, join(output_dir, f'{v["copy_path"]}.mp4'))

        v["yaml_dict"].pop("extraction_metadata", None)
        with open(f'{join(output_dir, v["copy_path"])}.yaml', "w") as f:
            yaml.safe_dump(v["yaml_dict"], f)


def aggregate_extract_results(input_dir, format, output_dir):
    """
    Aggregate results to one folder and generate index file (moseq2-index.yaml).

    Args:
    input_dir (str): path to base directory containing all session folders
    format (str): string format for metadata to use as the new aggregated filename
    output_dir (str): name of the directory to create and store all results in
    mouse_threshold (float): threshold value of mean frame depth to include session frames

    Returns:
    indexpath (str): path to generated index file including all aggregated session information.
    """

    h5s, dicts, _ = recursive_find_h5s(input_dir)

    not_in_output = lambda f: not exists(join(output_dir, basename(f)))
    complete = lambda d: d["complete"] and not d["skip"]

    def filter_h5(args):
        """remove h5's that should be skipped or extraction wasn't complete"""
        _dict, _h5 = args
        return complete(_dict) and not_in_output(_h5) and ("sample" not in _dict)

    # load in all of the h5 files, grab the extraction metadata, reformat to make nice 'n pretty
    # then stage the copy
    to_load = list(filter(filter_h5, zip(dicts, h5s)))

    loaded = load_extraction_meta_from_h5s(to_load)

    manifest = build_manifest(loaded, format=format)

    copy_manifest_results(manifest, output_dir)

    print("Results successfully aggregated in", output_dir)

    indexpath = generate_index(output_dir, join(input_dir, "moseq2-index.yaml"))

    print(f"Index file path: {indexpath}")
    return indexpath


def generate_index_from_agg_res(input_dir):
    """
    Generate index file from aggregated results folder.

    Args:
    input_dir (str): path to aggregated results folder
    """

    # find the yaml files
    yaml_paths = glob(os.path.join(input_dir, "*.yaml"))
    # setup pca path
    pca_path = os.path.join(os.path.dirname(input_dir), "_pca", "pca_scores.h5")
    if os.path.exists(pca_path):
        # point pca_path to pca scores you have
        index_data = {
            "files": [],
            "pca_path": pca_path,
        }
    else:
        index_data = {
            "files": [],
            "pca_path": "",
        }

    for p in yaml_paths:
        temp_yaml = read_yaml(p)
        file_dict = {
            "group": "default",
            "metadata": temp_yaml["metadata"],
            "path": [p[:-4] + "h5", p],
            "uuid": temp_yaml["uuid"],
        }
        index_data["files"].append(file_dict)

    # find output filename
    output_file = os.path.join(os.path.dirname(input_dir), "moseq2-index.yaml")

    # write out index yaml
    with open(output_file, "w") as f:
        yaml.safe_dump(index_data, f)


def write_frames(
    filename,
    frames,
    threads=6,
    fps=30,
    pixel_format="gray16le",
    codec="ffv1",
    close_pipe=True,
    pipe=None,
    frame_dtype="uint16",
    slices=24,
    slicecrc=1,
    frame_size=None,
    get_cmd=False,
):
    """
    Write frames to avi file using the ffv1 lossless encoder

    Args:
    filename (str): path to file to write to.
    frames (np.ndarray): frames to write
    threads (int): number of threads to write video
    fps (int): frames per second
    pixel_format (str): format video color scheme
    codec (str): ffmpeg encoding-writer method to use
    close_pipe (bool): indicates to close the open pipe to video when done writing.
    pipe (subProcess.Pipe): pipe to currently open video file.
    frame_dtype (str): indicates the data type to use when writing the videos
    slices (int): number of frame slices to write at a time.
    slicecrc (int): check integrity of slices
    frame_size (tuple): shape/dimensions of image.
    get_cmd (bool): indicates whether function should return ffmpeg command (instead of executing)

    Returns:
    pipe (subProcess.Pipe): indicates whether video writing is complete.
    """

    # we probably want to include a warning about multiples of 32 for videos
    # (then we can use pyav and some speedier tools)
    if not frame_size and type(frames) is np.ndarray:
        frame_size = "{0:d}x{1:d}".format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = "{0:d}x{1:d}".format(frames[0], frames[1])

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "fatal",
        "-framerate",
        str(fps),
        "-f",
        "rawvideo",
        "-s",
        frame_size,
        "-pix_fmt",
        pixel_format,
        "-i",
        "-",
        "-an",
        "-vcodec",
        codec,
        "-threads",
        str(threads),
        "-slices",
        str(slices),
        "-slicecrc",
        str(slicecrc),
        "-r",
        str(fps),
        filename,
    ]

    if get_cmd:
        return command

    if not pipe:
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in tqdm(
        range(frames.shape[0]), disable=True, desc=f"Writing frames to {filename}"
    ):
        pipe.stdin.write(frames[i].astype(frame_dtype).tostring())

    if close_pipe:
        pipe.communicate()
        return None
    else:
        return pipe


def convert_raw_to_avi(
    input_file, output_file, chunk_size, fps, delete, threads, mapping
):
    """
    compress a raw depth file into an avi file (with depth values) that is 8x smaller.

    Args:
    input_file (str): Path to depth file to convert
    output_file (str): Path to output avi file
    chunk_size (int): Size of frame chunks to iteratively process
    fps (int): frame rate.
    delete (bool): Delete the original depth file if True.
    threads (int): Number of threads used to encode video.
    mapping (str or int): Indicate which video stream to from the inputted file

    Returns:
    """

    if output_file is None:
        base_filename = splitext(basename(input_file))[0]
        output_file = join(dirname(input_file), f"{base_filename}.avi")

    vid_info = get_movie_info(input_file, mapping=mapping)
    frame_batches = gen_batch_sequence(vid_info["nframes"], chunk_size, 0)
    video_pipe = None

    for batch in tqdm(frame_batches, desc="Encoding batches"):
        frames = load_movie_data(input_file, batch, mapping=mapping)
        video_pipe = write_frames(
            output_file,
            frames,
            pipe=video_pipe,
            close_pipe=False,
            threads=threads,
            fps=fps,
        )

    if video_pipe:
        video_pipe.communicate()

    for batch in tqdm(frame_batches, desc="Checking data integrity"):
        raw_frames = load_movie_data(input_file, batch, mapping=mapping)
        encoded_frames = load_movie_data(output_file, batch, mapping=mapping)

        if not np.array_equal(raw_frames, encoded_frames):
            raise RuntimeError(
                f"Raw frames and encoded frames not equal from {batch[0]} to {batch[-1]}"
            )

    print("Encoding successful")

    if delete:
        print("Deleting", input_file)
        os.remove(input_file)


@filter_warnings
def download_flip(config_file, output_dir, selected_flip=None):
    """
    Download and save flip classifiers.

    Args:
    config_file (str): path to config file
    output_dir (str): path to directory to save classifier in.
    selected_flip (int or str): int: index of desired flip classifier; str: path to flip file

    Returns:
    None
    """

    flip_files = {
        "large mice with fibers (K2)": "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_largemicewithfiber.pkl",
        "adult male c57s (K2)": "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_c57_10to13weeks.pkl",
        "mice with Inscopix cables (K2)": "https://storage.googleapis.com/flip-classifiers/flip_classifier_k2_inscopix.pkl",
        "adult male c57s (Azure)": "https://moseq-data.s3.amazonaws.com/flip-classifier-azure-temp.pkl",
    }

    key_list = list(flip_files)

    if selected_flip is None:
        for idx, (k, v) in enumerate(flip_files.items()):
            print(f"[{idx}] {k} ---> {v}")
    else:
        selected_flip = key_list[selected_flip]

    # prompt for user selection if not already inputted
    while selected_flip is None:
        try:
            selected_flip = key_list[int(input("Enter a selection "))]
        except ValueError:
            print("Please enter a valid number listed above")
            continue

    if not exists(output_dir):
        os.makedirs(output_dir)

    selection = flip_files[selected_flip]

    output_filename = join(output_dir, basename(selection))

    urllib.request.urlretrieve(selection, output_filename)
    print("Successfully downloaded flip file to", output_filename)

    # Update the config file with the latest path to the flip classifier
    try:
        config_data = read_yaml(config_file)
        config_data["flip_classifier"] = output_filename

        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)
    except Exception as e:
        print("Could not update configuration file flip classifier path")
        print("Unexpected error:", e)


def read_frames_raw(
    filename,
    frames=None,
    frame_size=(512, 424),
    bit_depth=16,
    movie_dtype="<u2",
    **kwargs,
):
    """
    Reads in data from raw binary file.

    Args:
    filename (string): name of raw data file
    frames (list or range): frames to extract
    frame_dims (tuple): wxh of frames in pixels
    bit_depth (int): bits per pixel (default: 16)
    movie_dtype (str): An indicator for numpy to store the piped ffmpeg-read video in memory for processing.

    Returns:
    chunk (numpy ndarray): nframes x h x w
    """

    vid_info = get_raw_info(filename, frame_size=frame_size, bit_depth=bit_depth)

    if vid_info["dims"] != frame_size:
        frame_size = vid_info["dims"]

    if type(frames) is int:
        frames = [frames]
    elif not frames or (type(frames) is range) and len(frames) == 0:
        frames = range(0, vid_info["nframes"])

    seek_point = np.maximum(0, frames[0] * vid_info["bytes_per_frame"])
    read_points = len(frames) * frame_size[0] * frame_size[1]

    dims = (len(frames), frame_size[1], frame_size[0])

    if type(filename) is tarfile.TarFile:
        tar_members = filename.getmembers()
        tar_names = [_.name for _ in tar_members]
        input_file = tar_members[tar_names.index("depth.dat")]
        with filename.extractfile(input_file) as f:
            f.seek(int(seek_point))
            chunk = f.read(int(len(frames) * vid_info["bytes_per_frame"]))
            chunk = np.frombuffer(chunk, dtype=np.dtype(movie_dtype)).reshape(dims)
    else:
        with open(filename, "rb") as f:
            f.seek(int(seek_point))
            chunk = np.fromfile(
                file=f, dtype=np.dtype(movie_dtype), count=read_points
            ).reshape(dims)

    return chunk


def load_movie_data(
    filename, frames=None, frame_size=(512, 424), bit_depth=16, **kwargs
):
    """
    Parse file extension and load the movie data into numpy array.

    Args:
    filename (str): Path to video.
    frames (int or list): Frame indices to read in to output array.
    frame_size (tuple): Video dimensions (nrows, ncols)
    bit_depth (int): Number of bits per pixel, corresponds to image resolution.
    kwargs (dict): Any additional parameters that could be required in read_frames_raw().

    Returns:
    frame_data (numpy.ndarray): Read video as numpy array. (nframes, nrows, ncols)
    """

    if type(frames) is int:
        frames = [frames]
    try:
        if type(filename) is tarfile.TarFile:
            frame_data = read_frames_raw(
                filename,
                frames=frames,
                frame_size=frame_size,
                bit_depth=bit_depth,
                **kwargs,
            )
        elif filename.lower().endswith(".dat"):
            frame_data = read_frames_raw(
                filename,
                frames=frames,
                frame_size=frame_size,
                bit_depth=bit_depth,
                **kwargs,
            )
        elif filename.lower().endswith(".avi"):
            frame_data = read_frames(filename, frames, frame_size=frame_size, **kwargs)

    except AttributeError as e:
        print("Error reading movie:", e)
        frame_data = read_frames_raw(
            filename,
            frames=frames,
            frame_size=frame_size,
            bit_depth=bit_depth,
            **kwargs,
        )

    return frame_data


def read_image(filename, scale=True, scale_key="scale_factor"):
    """
    Load image data

    Args:
    filename (str): path to output file
    scale (bool): flag that indicates whether to scale image
    scale_key (str): indicates scale factor.

    Returns:
    image (numpy.ndarray): loaded image
    """

    with tifffile.TiffFile(filename) as tif:
        tmp = tif

    image = tmp.asarray()

    if scale:
        image_desc = json.loads(tmp.pages[0].tags["image_description"].as_str()[2:-1])

        try:
            scale_factor = int(image_desc[scale_key])
        except ValueError:
            scale_factor = ast.literal_eval(image_desc[scale_key])

        if type(scale_factor) is int:
            image = image / scale_factor
        elif type(scale_factor) is tuple:
            iinfo = np.iinfo(image.dtype)
            image = image.astype("float32") / iinfo.max
            image = image * (scale_factor[1] - scale_factor[0]) + scale_factor[0]

    return image