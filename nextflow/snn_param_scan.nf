
def ratmoseq_env = "/home/jal5475/.miniconda/envs/sizenorm"

// supply path to config file that contains the parameter scan values
params.config_file = "/n/groups/datta/jlove/data/rat_seq/rat_seq_paper/ratmoseq-sizenorm/configs/01-sizenorm_param_scan.toml"
// the stage is one of the sections defined in the scan config files 
params.stage = 1
params.seed = "0"
params.stageList = "${params.stage}".tokenize(',') as List

// this creates the parameter scan grid based on the stage
process create_grid {
    executor 'local'
    conda ratmoseq_env

    input:
    val config_file
    val stage

    output:
    stdout emit: file_name

    script:
    """
    python /n/groups/datta/jlove/data/rat_seq/rat_seq_paper/ratmoseq-sizenorm/scripts/07-batch-scan-hpparams.py \
    /n/groups/datta/jlove/data/rat_seq/rat_seq_paper/ratmoseq-sizenorm/configs/00-sizenorm_training_template.toml \
    $config_file --stage $stage --reset-run --seed $params.seed
    """
}

// this trains a size norm model for each parameter combination from the 
// previous step
process run_grid {
    executor 'slurm'
    label 'gpu_quad'
    memory 15.GB
    time { 4.h + (task.attempt - 1) * 3.h }
    maxRetries 5
    conda ratmoseq_env

    input:
    val config_file

    output:
    val config_file

    script:
    """
    python /n/groups/datta/jlove/data/rat_seq/rat_seq_paper/ratmoseq-sizenorm/scripts/03-train-size-norm.py $config_file --checkpoint
    """
}

workflow {
    configs = create_grid(Channel.value(params.config_file), Channel.fromList(params.stageList))
    configs = configs.map { file(it.trim()).readLines() }
        .flatten()

    run_grid(configs).view()
}
