"""
CLI for extracting the depth data.
"""

import os
import click
from ruamel.yaml import YAML
from tqdm.auto import tqdm
from copy import deepcopy
from ratmoseq_extract.extract import (
    run_extraction,
    run_local_batch_extract,
    run_slurm_batch_extract,
)
from ratmoseq_extract.io import (
    read_yaml,
    recursive_find_unextracted_dirs,
    generate_index,
    aggregate_extract_results,
    generate_index_from_agg_res,
    convert_raw_to_avi,
    download_flip,
)

orig_init = click.core.Option.__init__

yaml = YAML(typ='safe', pure=True)

def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


# from https://stackoverflow.com/questions/46358797/
# python-click-supply-arguments-and-options-from-a-configuration-file
def command_with_config(config_file_param_name):
    """
    Override default CLI variables with the values contained within the config.yaml being passed.

    Args:
    config_file_param_name (str): path to config file.

    Returns:
    custom_command_class (function): decorator function to update click.Command parameters with the config_file
    parameter values.
    """

    class custom_command_class(click.Command):

        def invoke(self, ctx):
            # grab the config file
            config_file = ctx.params[config_file_param_name]
            param_defaults = {p.human_readable_name: p.default for p in self.params
                              if isinstance(p, click.core.Option)}
            param_defaults = {k: tuple(v) if type(v) is list else v for k, v in param_defaults.items()}
            param_cli = {k: tuple(v) if type(v) is list else v for k, v in ctx.params.items()}

            if config_file is not None:

                config_data = read_yaml(config_file)
                # set config_data['output_file'] ['output_dir'] ['input_dir'] to None to avoid overwriting previous files
                # assuming users would either input their own paths or use the default path
                config_data['input_dir'] = None
                config_data['output_dir'] = None
                config_data['output_file'] = None
                
                # find differences btw config and param defaults
                diffs = set(param_defaults.items()) ^ set(param_cli.items())

                # combine defaults w/ config data
                combined = {**param_defaults, **config_data}

                # update cli params that are non-default
                keys = [d[0] for d in diffs]
                for k in set(keys):
                    combined[k] = ctx.params[k]

                ctx.params = combined
                
                # add new parameters to the original config file
                config_data = read_yaml(config_file)
                
                # remove flags from combined so the flag values in config.yaml won't get overwritten
                flag_list = ['manual_set_depth_range', 'use_plane_bground', 'progress_bar', 'delete', 'compute_raw_scalars', 'skip_completed', 'skip_checks', 'get_cmd', 'run_cmd']
                combined = {k:v for k, v in combined.items() if k not in flag_list}
                # combine original config data and the combined params prioritizing the combined
                config_data = {**config_data, **combined}
                # with open(config_file, 'w') as f:
                #     yaml.safe_dump(config_data, f)

            return super().invoke(ctx)

    return custom_command_class

click.core.Option.__init__ = new_init


@click.group()
@click.version_option()
def cli():
    pass


def common_avi_options(function):
    """
    Decorator function for grouping shared video processing parameters.

    Args:
    function: Function to add enclosed parameters to as click options.

    Returns:
    function: Updated function including shared parameters.
    """

    function = click.option(
        "-o",
        "--output-file",
        type=click.Path(),
        default=None,
        help="Path to output file",
    )(function)
    function = click.option(
        "-b", "--chunk-size", type=int, default=500, help="Chunk size"
    )(function)
    function = click.option("--fps", type=float, default=30, help="Video FPS")(function)
    function = click.option(
        "--delete", is_flag=True, help="Delete raw file if encoding is sucessful"
    )(function)
    function = click.option(
        "-t", "--threads", type=int, default=8, help="Number of threads for encoding"
    )(function)
    function = click.option(
        "-m",
        "--mapping",
        type=str,
        default="DEPTH",
        help="Ffprobe stream selection variable. Default: DEPTH",
    )(function)

    return function


def extract_options(function):
    """
    Decorator function for grouping shared extraction prameters.

    Args:
    function : Function to add enclosed parameters to as click options.

    Returns:
    function: Updated function including shared parameters.
    """
    
    function = click.option("--config-file", type=click.Path())(function)
    function = click.option(
        "--crop-size",
        "-c",
        default=(256, 256),
        type=(int, int),
        help="Width and height of cropped mouse image",
    )(function)
    function = click.option(
    "--bg-depth-range",
    "-c",
    default=(900, 1000),
    type=(int, int),
    help="Width and height of cropped mouse image",
    )(function)
    function = click.option(
        "--num-frames",
        "-n",
        default=None,
        type=int,
        help="Number of frames to extract. Will extract full session if set to None.",
    )(function)
    function = click.option(
        "--min-height",
        default=10,
        type=int,
        help="Min mouse height threshold from floor (mm)",
    )(function)
    function = click.option(
        "--max-height",
        default=310,
        type=int,
        help="Max mouse height threshold from floor (mm)",
    )(function)
    function = click.option(
        "--detected-true-depth",
        default="auto",
        type=str,
        help='Option to override automatic depth estimation during extraction. \
This is only a debugging parameter, for cases where dilate_iterations > 1, otherwise has no effect. Either "auto" or an int value.',
    )(function)
    function = click.option(
        "--compute-raw-scalars",
        is_flag=True,
        help="Compute scalar values from raw cropped frames.",
    )(function)
    function = click.option(
        "--flip-classifier",
        default=None,
        help="path to the flip classifier used to properly orient the mouse (.pkl file)",
    )(function)
    function = click.option(
        "--flip-classifier-smoothing",
        default=None,
        type=int,
        help="Number of frames to smooth flip classifier probabilities",
    )(function)
    function = click.option(
        "--use-cc",
        default=True,
        type=bool,
        help="Extract features using largest connected components.",
    )(function)
    function = click.option(
        "--tail-ksize",
        default=15,
        type=int,
        help="Tail filter kernel size",
    )(function)
    function = click.option(
        "--dilation-ksize",
        default=5,
        type=int,
        help="Dilation kernel size",
    )(function)
    function = click.option(
        "--chunk-overlap",
        default=0,
        type=int,
        help="Frames overlapped in each chunk. Useful for cable tracking",
    )(function)
    function = click.option(
        "--write-movie",
        default=True,
        type=bool,
        help="Write a results output movie including an extracted mouse",
    )(function)
    function = click.option(
        "--frame-dtype",
        default="int",
        type=click.Choice(["uint8", "uint16", "int"]),
        help="Data type for processed frames",
    )(function)
    function = click.option(
        "--movie-dtype",
        default="<i2",
        help="Data type for raw frames read in for extraction",
    )(function)
    function = click.option(
        "--pixel-format",
        default="gray16le",
        type=str,
        help="Pixel format for reading in .avi and .mkv videos",
    )(function)
    function = click.option(
        "--model-smoothing-clips",
        default=(0, 0),
        type=(float, float),
        help="Model smoothing clips",
    )(function)
    function = click.option(
        "--frame-trim",
        default=(0, 0),
        type=(int, int),
        help="Frames to trim from beginning and end of data",
    )(function)
    function = click.option(
        "--compress",
        default=False,
        type=bool,
        help="Convert .dat to .avi after successful extraction",
    )(function)
    function = click.option(
        "--compress-chunk-size",
        type=int,
        default=500,
        help="Chunk size for .avi compression",
    )(function)
    function = click.option(
        "--compress-threads", type=int, default=3, help="Number of threads for encoding"
    )(function)
    function = click.option(
        "--skip-completed",
        is_flag=True,
        help="Will skip the extraction if it is already completed.",
    )(function)
    function = click.option(
        "--sam2-checkpoint",
        type=click.Path(),
        default=None,
        help="Path to SAM2 checkpoint file",
    )(function)
    function = click.option(
        "--dlc-filename", type=str, default=None, help="DLC filename for SAM2"
    )(function)
    function = click.option(
        "--num-frames", type=int, default=None, help="Number of frames to extract"
    )(function)
    function = click.option(
        "--outputdir", type=str, default='proc', help="Output directory for processed data"
    )(function)

    return function


@cli.command(
    name="extract",
    cls=command_with_config("config_file"),
    help="Processes raw input depth recordings to output a cropped and oriented"
    "video of the mouse and saves the output+metadata to h5 files in the given output directory.",
)
@click.argument("input-file", type=click.Path(exists=True, resolve_path=False))
@click.option(
    "--cluster-type",
    type=click.Choice(["local", "slurm"]),
    default="local",
    help="Platform to train models on",
)
@common_avi_options
@extract_options
def extract(input_file, **config_data):

    run_extraction(
        input_file, config_data
    )


@cli.command(
    name="batch-extract",
    cls=command_with_config("config_file"),
    help="Batch processes " "all the raw depth recordings located in the input folder.",
)
@click.argument("input-folder", type=click.Path(exists=True, resolve_path=False))
@common_avi_options
@extract_options
@click.option(
    "--extensions",
    default=[".dat"],
    type=str,
    help="File extension of raw data",
    multiple=True,
)
@click.option(
    "--skip-checks",
    is_flag=True,
    help="Flag: skip checks for the existance of a metadata file",
)
@click.option(
    "--extract-out-script",
    type=click.Path(),
    default="extract_out.sh",
    help="Name of bash script file to save extract commands.",
)
@click.option(
    "--cluster-type",
    type=click.Choice(["local", "slurm"]),
    default="local",
    help="Platform to train models on",
)
@click.option(
    "--prefix",
    type=str,
    default="",
    help="Batch command string to prefix model training command (slurm only).",
)
@click.option(
    "--ncpus", "-c", type=int, default=1, help="Number of cores to use in extraction"
)
@click.option("--memory", type=str, default="5GB", help="RAM (slurm only)")
@click.option("--wall-time", type=str, default="3:00:00", help="Wall time (slurm only)")
@click.option(
    "--partition", type=str, default="short", help="Partition name (slurm only)"
)
@click.option(
    "--get-cmd", is_flag=True, default=True, help="Print scan command strings."
)
@click.option("--run-cmd", is_flag=True, help="Run scan command strings.")
def batch_extract(
    input_folder,
    output_dir,
    skip_completed,
    num_frames,
    extensions,
    skip_checks,
    **config_data,
):

    # check if there is a config file
    config_file = config_data.get("config_file")
    if not config_file:
        # Add message to tell the users to specify a config file
        print(
            "Command not run. Please specified a config file using --config-file flag."
        )
        return

    # Add message to tell the users to specify a config file
    to_extract = []
    for ex in extensions:
        to_extract.extend(
            recursive_find_unextracted_dirs(
                input_folder,
                extension=ex,
                skip_checks=True if ex in (".tgz", ".tar.gz") else skip_checks,
                yaml_path=os.path.join(output_dir, "results_00.yaml"),
            )
        )

    # Add message when all sessions are extracted
    if len(to_extract) == 0:
        print(
            'No session to be extracted. If you want to re-extract the data, please add "--skip-checks"'
        )
        return

    if config_data["cluster_type"] == "local":
        # the session specific config doesn't get generated in session proc file
        # session specific config direct used in config_data dictionary in extraction from extract_command function
        run_local_batch_extract(to_extract, config_file, skip_completed)
    else:
        # add paramters to config
        config_data["session_config_path"] = (
            read_yaml(config_file).get("session_config_path", "")
            if config_file is not None
            else ""
        )
        config_data["config_file"] = os.path.abspath(config_file)
        config_data["output_dir"] = output_dir
        config_data["skip_completed"] = skip_completed
        config_data["num_frames"] = num_frames
        config_data["extensions"] = extensions
        config_data["skip_checks"] = skip_checks
        # run slurm extract will generate a config.yaml in session proc file for slurm
        run_slurm_batch_extract(input_folder, to_extract, config_data, skip_completed)


@cli.command(
    name="download-flip-file",
    help="Downloads Flip-correction model that helps with orienting the mouse during extraction.",
)
@click.argument(
    "config-file",
    type=click.Path(exists=True, resolve_path=False),
    default="config.yaml",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=os.getcwd(),
    help="Output directory for downloaded flip flie",
)
def download_flip_file(config_file, output_dir):

    download_flip(config_file, output_dir)


@cli.command(
    name="generate-config",
    help="Generates a configuration file (config.yaml) that holds editable options for extraction parameters.",
)
@click.option("--output-file", "-o", type=click.Path(), default="config.yaml")
@click.option(
    "--camera-type",
    default="azure",
    type=str,
    help="specify the camera type (k2 or azure), default is azure",
)
def generate_config(output_file, camera_type):

    objs = extract.params
    params = {tmp.name: tmp.default for tmp in objs if not tmp.required}
    if camera_type == "azure":
        # params["bg_roi_depth_range"] = [550, 650]
        params["spatial_filter_size"] = [5]
        params["tail_filter_size"] = [15, 15]
        params["crop_size"] = [256, 256]
        params["camera_type"] = "azure"

    with open(output_file, "w") as f:
        yaml.dump(params, f)  # Dump the params into the file
        
    print("Successfully generated config file in base directory.")


@cli.command(
    name="generate-index",
    help="Generates an index file (moseq2-index.yaml) that contains all extracted session metadata.",
)
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(),
    default=os.getcwd(),
    help="Directory to find h5 files",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "moseq2-index.yaml"),
    help="Location for storing index",
)
def generate_index(input_dir, output_file):

    output_file = generate_index(input_dir, output_file)

    if output_file is not None:
        print(f"Index file: {output_file} was successfully generated.")


@cli.command(
    name="aggregate-results",
    help="Copies all extracted results (h5, yaml, mp4) files from all extracted sessions to a new directory for modeling and analysis",
)
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(),
    default=os.getcwd(),
    help="Directory to find h5 files",
)
@click.option(
    "--format",
    "-f",
    type=str,
    default="{start_time}_{session_name}_{subject_name}",
    help="New file name formats from resepective metadata",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "aggregate_results/"),
    help="Location for storing all results together",
)
def aggregate_extract_results(input_dir, format, output_dir, mouse_threshold):

    aggregate_extract_results(input_dir, format, output_dir)


@cli.command(
    name="agg-to-index",
    help="Generate an index file from aggregated results with default as group names",
)
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(),
    default=os.path.join(os.getcwd(), "aggregate_results"),
    help="Directory for aggregated results folder",
)
def agg_to_index(input_dir):

    generate_index_from_agg_res(input_dir)


@cli.command(
    name="convert-raw-to-avi",
    help="Loss less compresses a raw depth file (dat) into an avi file that is 8x smaller.",
)
@click.argument("input-file", type=click.Path(exists=True, resolve_path=False))
@common_avi_options
def convert_raw_to_avi(
    input_file, output_file, chunk_size, fps, delete, threads, mapping
):

    convert_raw_to_avi(
        input_file, output_file, chunk_size, fps, delete, threads, mapping
    )


if __name__ == "__main__":
    cli()
