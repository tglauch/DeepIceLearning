import os

tdict={'kta': '2:00:00', 'long': '48:00:00', 'gpu':'168:00:00'}
def make_condor(executable, request_gpus, request_memory, requirements, addpath,
                arguments, thisfolder):
    submit_info = '\
            executable   = {folder}/{script} \n\
            universe     = vanilla  \n\
            request_gpus = {gpu} \n\
            request_memory = {mem}GB \n\
            requirements = {req} \n\
            log          = {addp}/{script}.log \n\
            output       = {addp}/{script}.out \n\
            error        = {addp}/{script}.err \n\
            stream_output = True \n\
            IWD = {folder} \n\
            arguments =  {args} \n\
            queue 1 \n '.format(script=executable,\
                                gpu=request_gpus,\
                                mem=request_memory,\
                                req=requirements,\
                                addp=addpath,\
                                args=arguments,\
                                folder=thisfolder)
    return submit_info


def make_slurm(executable, request_gpus, request_memory, condor_folder, file_location,
               arguments, thisfolder, exclude='nothing', partition='gpu', ex_type='bash', log_name=None):

    if log_name == None:
	log_name = executable
    if exclude != '':
        exclude_node = '#SBATCH --exclude {} \n'.format(exclude)
    else:
        exclude_node = "" 

# Please do not ident!!!
    submit_info = '#!/usr/bin/env bash\n\
#SBATCH --time={time}\n\
#SBATCH --partition={part}\n\
#SBATCH --gres=gpu:{req_gpus}\n\
#SBATCH --mem={req_mem} \n\
#SBATCH --error={cond_fold}/{log_name}.err\n\
#SBATCH --output={cond_fold}/{log_name}.out\n\
{excl_node}\
\n\
{ex_type} {thisfolder}/submit_scripts/{script} {args} \n'.format(
        script = executable,\
        req_gpus = request_gpus,\
        req_mem = int(request_memory),\
        cond_fold = condor_folder,\
        args = arguments,\
        thisfolder = thisfolder,\
        excl_node = exclude_node,\
	part=partition,
	time=tdict[partition],
        log_name=log_name,
	ex_type=ex_type)

    return submit_info


def make_bsub(executable, request_memory, condor_folder, thisfolder,
              arguments, apply_test=False, request_cpus=12, cfg_file= None,
              save_path = None):
    submit_info = "#!/usr/bin/env zsh\n\
#BSUB -J {script}.job\n\
#BSUB -W 48:00\n\
#BSUB -M {mem_request}\n\
#BSUB -n {request_cpus}\n\
#BSUB -o {cond_fold}/{script}.out\n\
#BSUB -e {cond_fold}/{script}.err\n\
#BSUB -a 'gpu openmp'\n\
#BSUB -R pascal\n\
#BSUB -P phys3b\n\
source /home/phys3b/Envs/keras_tf/bin/activate\n\
export CUDA_VISIBLE_DEVICES=`/home/phys3b/etc/check_gpu.py 2`\n\
if [ '$CUDA_VISIBLE_DEVICES' = '-1' ];\n\
then\
    echo '##### GPUs busy. Restart job later.' exit 1\n\
else\
    echo 'Found free GPU devices :'\n\
    echo 'CUDA_VISIBLE_DEVICES =  $CUDA_VISIBLE_DEVICES '\n\
fi\n\
nvidia-smi\n\
python {NN_recofolder}/{script} {args}\n".\
                format(script = executable,\
                       request_cpus = request_cpus,
                       cond_fold=condor_folder,
                       args=arguments,
                       NN_recofolder=thisfolder,
                       mem_request=request_memory)
    if apply_test:
        submit_info += "python {NN_recofolder}/Apply_Model.py --main_config\
        {cfg} --folder {save_path} \n".\
                format(cfg = cfg_file,
                       save_path = save_path,
                       NN_recofolder = thisfolder)

    return submit_info



