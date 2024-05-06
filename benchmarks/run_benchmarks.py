PYTHONPATH="/data/aaron/experimental/fms-hf-tuning"
PYTHON_TEMPLATE_PREFIX = "PYTHONPATH={root_dir} CUDA_VISIBLE_DEVICES=0 python {root_dir}/tuning/sft_trainer.py"
ACCELERATE_TEMPLATE_PREFIX = "PYTHONPATH={root_dir} CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file {root_dir}/fixtures/accelerate_fsdp_defaults.yaml --num_processes={num_processes} --main_process_port=29500 {root_dir}/tuning/sft_trainer.py"

import yaml
import subprocess
import json
import pandas as pdza

with open("configurations.yaml") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

def args_to_str(args_dict):
    return ' '.join([f"--{key} {val}" if val else f"--{key}" for key, val in args_dict.items()])

experiment_stats = {}
for variant, variant_conf in config['variants'].items():
    print(variant)

    # 1.  Prep experiment variant args
    common_args_dict = {key: val for key, val in config['common'].items() if key not in variant_conf.keys()}
    common_args = args_to_str(config['common'])
    num_processes = variant_conf.pop('num_processes')
    var_arg = args_to_str(variant_conf)

    # 2. Prep the relevant prefix to run (python or accelerate for distributed)
    prefix = PYTHON_TEMPLATE_PREFIX.format(root_dir=PYTHONPATH)
    if num_processes > 1:
        prefix = ACCELERATE_TEMPLATE_PREFIX.format(root_dir=PYTHONPATH, num_processes=num_processes)

    # 3. assemble the variant run command
    run_cmd = f"{prefix} {common_args} {var_arg}"

    # 4. run experiment variant as a subprocess
    result = subprocess.run([run_cmd], shell=True, capture_output=False, text=True)

    # 5. store the results for collation
    with open(f"{variant_conf['output_dir']}/cli.txt", 'w') as f:
        f.write(run_cmd)
    
    if result.stdout:
        with open(f"{variant_conf['output_dir']}/out.txt", 'w') as f:
            f.write(result.stdout)

    stats = json.load(open(f"{variant_conf['output_dir']}/stats.json", 'r'))
    experiment_stats[variant] = stats['train_tokens_per_second']

# Get a summary of all the runs
vanilla_df = pd.Series({key:val for key, val in experiment_stats.items() if 'unsloth' not in key}, name='throughput (tokens/sec)')
unsloth_df = pd.Series({key:val for key, val in experiment_stats.items() if 'unsloth' in key}, name='throughput (tokens/sec)  - Unsloth')
unsloth_df.index = unsloth_df.index.str.replace('unsloth_', '')
result_df = pd.concat([vanilla_df, unsloth_df], axis=1)
result_df['% improvement'] = (result_df["Unsloth_throughput (tokens/sec)"] - result_df["throughput (tokens/sec)"])/result_df["throughput (tokens/sec)"]
result_df.to_csv('./results/summary.csv')
