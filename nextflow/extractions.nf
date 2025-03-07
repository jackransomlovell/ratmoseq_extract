// define conda environment paths - set to your own paths
def ratmoseq_env = "/home/jal5475/.miniconda/envs/ratmoseq_extract"
def deeplabcut_env = "/home/jal5475/.miniconda/envs/DEEPLABCUT"
def sizenorm_env = "/home/jal5475/.miniconda/envs/sizenorm"

// set workflow parameters - these can be modified from the command line
params.data_path = "/n/groups/datta/jlove/data/rat_seq/data_managment/habitat/fmr1"
params.size_norm_name = "size_norm_v1"
params.keypoint_name = "ir_clippedDLC_resnet50_KeypointMoSeqDLCOct18shuffle1_50000"
params.dlc_config = "/n/groups/datta/jlove/data/rat_seq/rat_seq_paper/keypoint_model/config-v2.yaml"
params.sam2_checkpoint = "/n/groups/datta/jlove/data/sam2/checkpoints/sam2.1_hiera_tiny.pt"
params.flip_classifier = "/n/groups/datta/jlove/data/rat_seq/rat_seq_paper/data/14weeks-flip.p"
// pytorch model for size normalization
params.snn_path = "/n/groups/datta/jlove/data/rat_seq/rat_seq_paper/sizenorm_training/models/04_template/stage_09/optim_model/model.pt"
//params.snn_path = "/n/groups/datta/win/longtogeny/size_norm/models/freeze_decoder_00/stage_09/7b96ec7e-f894-4391-8c39-f0cb8d7dd516/model.pt"

// name of the moseq2-extract config.yaml file used for extractions
// saved in /n/groups/datta/win/longtogeny/data/extractions/
params.config_name = "config-2024-04-25"
// folder to save extractions
params.proc_name = "proc-2024-04-25"
// set to 1 to force extraction of all files, even if they have already been extracted. 0 for only unextracted files
params.force_extract = 0


// find files that need ir to be clipped
process find_ir_clipped_files {
    executor "local"
    conda ratmoseq_env
    
    output:
    path "ir_clipped_files.txt"

    script:
    """
    #!/bin/env python
    from pathlib import Path
    from ratmoseq_extract.io import get_depth_files, no_ir_clipped

    path = Path("${params.data_path}")
    files = get_depth_files(path)
    files = list(filter(no_ir_clipped, files))
    files = files[:3]
    with open("ir_clipped_files.txt", "w") as f:
        for file in files:
            f.write(str(file) + "\\n")
    """
}

process clip_ir {
    label "short"
    memory 16.GB
    cpus 1
    time { 20.m * task.attempt }
    maxRetries 1
    conda ratmoseq_env

    input:
    val ir_file

    script:
    """
    #!/bin/env python
    from pathlib import Path
    from ratmoseq_extract.io import clip_ir

    file_path = Path("${ir_file}".strip())
    clip_ir(file_path, file_path.parent / "ir_clipped.avi")
    """
}

// find files that need keypoints
process find_keypointable_files {
    executor "local"
    conda ratmoseq_env

    output:
    path "keypointable_files.txt"

    script:
    """
    #!/bin/env python
    from pathlib import Path
    from ratmoseq_extract.io import get_depth_files, no_ir_clipped, no_keypoints

    path = Path("${params.data_path}")
    files = get_depth_files(path)
    files = list(filter(lambda x: not no_ir_clipped(x), files))
    files = list(filter(lambda x: no_keypoints(x, "${params.keypoint_name}"), files))
    files = files[:3]

    with open("keypointable_files.txt", "w") as f:
        for file in files:
            f.write(str(file) + "\\n")
    """
}

process extract_keypoints {
    label "gpu"
    memory 16.GB
    time { 20.m * task.attempt }
    maxRetries 2
    conda deeplabcut_env 

    input:
    val keypointable_files

    output:
    val keypointable_files

    script:
    """
    #!/bin/env python
    import deeplabcut 
    
    videos = "${keypointable_files}".strip()[1:-1].split(",")
    videos = [v.strip() for v in videos]
    
    config_path = "${params.dlc_config}"
    deeplabcut.analyze_videos(config_path, videos, videotype=".avi")
    deeplabcut.filterpredictions(config_path, videos, videotype=".avi")
    deeplabcut.create_labeled_video(config_path, videos, filtered=True, pcutoff=0.3)
    """
}

process find_extractable_files {
    executor "local"
    conda ratmoseq_env 

    output:
    path "extractable_files.txt"

    script:
    """
    #!/bin/env python
    from pathlib import Path
    from ratmoseq_extract.io import get_depth_files
    from ratmoseq_extract.io import not_extracted

    path = Path("${params.data_path}")
    files = get_depth_files(path)
    files = list(filter(not_extracted, files))
    files = files[:3]
    with open("extractable_files.txt", "w") as f:
        for file in files:
            f.write(str(file) + "\\n")
    """
}

// runs moseq2-extract on each depth file
process extract {
    label "gpu_quad"
    memory 24.GB
    time { 60.m * task.attempt }
    maxRetries 2
    conda ratmoseq_env

    input:
    val depth_file

    output:
    val depth_file

    script:
    """
    #!/bin/bash
    {
    ratmoseq-extract extract "${depth_file}" --sam2-checkpoint "${params.sam2_checkpoint}" --dlc-filename "${params.keypoint_name}.csv" --use-bground True --bground-type plane
    } || {
    echo "Extract command for ${depth_file} did not work"
    }
    """
}

// compresses the original depth files to avi format, saving 10x space
process compress {
    label "short"
    cpus 1
    memory 13.GB
    time { 75.m * task.attempt }
    maxRetries 1
    conda moseq_env

    input:
    val depth_file

    script:
    """
    moseq2-extract convert-raw-to-avi "${depth_file}" --delete || true
    """
}

// process to find extracted files that need to be size normalized
// this is run separately from the extraction process to allow for de-synchronization between the two
// for example, running a new size-norm model on previously extracted files
process find_files_to_normalize {
    executor "local"
    conda ratmoseq_env

    output:
    path "files_to_normalize.txt"

    script:
    """
    #!/bin/env python
    from pathlib import Path
    from ratmoseq_extract.io import get_depth_files#, hasnt_key
    import h5py

    def hasnt_key(file):
        with h5py.File(file, "r") as f:
            if "${params.size_norm_name}" not in f:
                return True
        return False

    files = get_depth_files(Path("${params.data_path}"))
    files = [f.parent / "proc/results_00.h5" for f in files]
    files = [f for f in files if f.exists()]

    with open("files_to_normalize.txt", "w") as f:
        for file in files:
            if hasnt_key(file):
                f.write(str(file) + "\\n")
    """
}

// run size normalization on each extracted file
process size_normalize {
    label "gpu_quad"
    memory 16.GB
    time {20.m * task.attempt }
    maxRetries 1
    conda sizenorm_env

    input:
    val file_collection

    script:
    """
    #!/bin/env python
    import torch
    from pathlib import Path
    from sizenorm.size_norm.apply import predict_and_save

    collection = "${file_collection}"
    collection = [s.strip() for s in collection.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load("${params.snn_path}", map_location=device)

    def predict(file):
        try:
            predict_and_save(Path(file), model, "${params.size_norm_name}", rescale=False, clean_noise=True)
        except Exception as e:
            print("Exception for file", file)
            print(e)
            print('---' * 3)

    for file in collection:
        predict(file)
    """
}

workflow {
    // Find files that need IR clipping
    files = find_ir_clipped_files()
    
    // Clip IR for first 3 files that need it
    files = files.map { it.readLines() }
        .flatten()
        .filter { it != "" && it != null && it != "\n" }

    clip_ir(files)

    // Find files that need keypoints
    files = find_keypointable_files()

    // Extract keypoints
    files = files.map { it.readLines() }
        .flatten()
        .filter { it != "" && it != null && it != "\n" }

    extract_keypoints(files)

    // Find files that need to be extracted
    files = find_extractable_files()

    // Extract
    files = files.map { it.readLines() }
        .flatten()
        .filter { it != "" && it != null && it != "\n" }

    extract(files)

    // Find files that need to be size normalized
    files = find_files_to_normalize()
    
    // Apply size normalization to files
    files = files.map { it.readLines() }
        .flatten()
        .filter { it != "" && it != null && it != "\n" }

    size_normalize(files)
}