import numpy as np
import cv2
import subprocess
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
import tarfile
import json
import warnings
import os
import tifffile
import click
import h5py
from pkg_resources import get_distribution
from os.path import exists, join, dirname, basename, splitext


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
        return yaml.safe_load(f)


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
    if config_data["use_tracking_model"]:
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
    extract_version = np.string_(get_distribution("moseq2-extract").version)
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
        return yaml.safe_load(f)


def write_image(
    filename, image, scale=True, scale_factor=None, frame_dtype="uint16", compress=0
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
        file, image.astype(frame_dtype), compress=compress, metadata=metadata
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
