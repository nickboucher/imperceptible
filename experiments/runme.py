import socket
import os  

def format_and_save(script, template, **kwargs):

    if socket.gethostname() != "v":
        mem_constraint = "--mem-per-gpu=13G"
        partition = "gpu"
        qos = "nopreemption"
    else:
        mem_constraint = "--mem=30G"
        partition = "t4v2"
        #qos = "nopreemption"
        qos = "normal"

    kwargs['mem_constraint'] = mem_constraint
    kwargs['partition'] = partition 
    kwargs['qos'] = qos 

    temp = template.format(**kwargs)

    sc = open(script, 'w')
    sc.write(temp)
    sc.close()

    print(temp)

slurm_script_template = (
    "#!/bin/bash\n"
    "#SBATCH --time=7-00:00            # time (DD-HH:MM)\n"
    "#SBATCH --cpus-per-task=2         # CPU cores/threads\n"
    "#SBATCH --partition={partition} # define the partition to run on\n"
    "#SBATCH --qos={qos} # define the partition to run on\n"
    "#SBATCH --gres=gpu:1                         # Number of GPUs (per node)\n"
    "#SBATCH --nodes=1\n"
    "#SBATCH --job-name={name}\n"
    "#SBATCH --output=/scratch/ssd001/home/iliash/code/keywordspotting_mixing_defense/TCResNet/log/slurm/flareon-%x-%A-%a.out # specify output file\n"
    "#SBATCH --error=/scratch/ssd001/home/iliash/code/keywordspotting_mixing_defense/TCResNet/log/error/flareon-%x-%A-%a.err  # specify error file\n"
    "#SBATCH {mem_constraint}\n"
    "export PATH=/pkgs/anaconda3/bin:$PATH\n"
    ". /pkgs/anaconda3/bin/activate {virt_env}\n"
    "echo \"$(uname -a)\"\n"
    "echo \"$(nvidia-smi)\"\n"
    "echo \"$(pwd)\"\n"
    "echo \"Running command\"\n"
    "echo \"$(which python)\"\n"
    "cd {exec_dir}\n"
)

commands = [l.strip()[2:] for l in open("commands.sh").readlines()]

scriptsfolder = "results"
os.makedirs(scriptsfolder, exist_ok=True)
expfolder = "/scratch/ssd001/home/iliash/code/imperceptible/experiments/"

for i, command in enumerate(commands):
    
    print(command) 

    _temp = slurm_script_template[:]
    prefix = f'PYTORCH_FAIRSEQ_CACHE="{expfolder}" TORCH_HOME="{expfolder}" CUDA_VISIBLE_DEVICES=0 '

    _temp += prefix + "python " + command 

    format_and_save(
        name = f"{i}.sh",
        script = f"{scriptsfolder}/{i}.sh",
        run_script = f"{scriptsfolder}/{i}.sh",
        template = _temp,
        virt_env= "/scratch/ssd001/home/iliash/virts/kws/",
        exec_dir= f"{expfolder}",
    )





