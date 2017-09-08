import os

def make_condor(request_gpus, request_memory, requirements, addpath,
                arguments, thisfolder):
    submit_info = '\
            executable   = {folder}/Neural_Network.py \n\
            universe     = vanilla  \n\
            request_gpus = {gpu} \n\
            request_memory = {mem}GB \n\
            requirements = {req} \n\
            log          = {addp}/condor.log \n\
            output       = {addp}/condor.out \n\
            error        = {addp}/condor.err \n\
            stream_output = True \n\
            getenv = True \n\
            IWD = {folder} \n\
            arguments =  {args} \n\
            queue 1 \n '.format(gpu=request_gpus, mem=request_memory,
                                req=requirements, addp=addpath,
                                args=arguments, folder=thisfolder)
    return submit_info


def make_slurm(request_gpus, request_memory, condor_folder, file_location,
               arguments, thisfolder, exclude=''):

    if exclude != '':
        exclude_node = '#SBATCH --exclude {} \n'.format(exclude)

    submit_info = '#!/usr/bin/env bash\n\
            #SBATCH --time=48:00:00\n\
            #SBATCH --partition=gpu\n\
            #SBATCH --gres=gpu:{0}\n\
            #SBATCH --mem={1} \n\
            #SBATCH --error={2}/condor.err\n\
            #SBATCH --output={2}/condor.out\n\
            {5}\
            \n\
            python {4}/Neural_Network.py {3} \n'.format(
        request_gpus, int(request_memory),
        condor_folder, arguments, thisfolder, exclude_node)

    return submit_info

def make_bsub(request_memory, condor_folder, thisfolder,
               arguments, request_cpus = 12):
    submit_info = "#!/usr/bin/env zsh\n\
            #BSUB -J trainNN.job\n\
            #BSUB -W 12:00\n\
            #BSUB -M {mem_request}\n\
            #BSUB -n {request_cpus}\n\
            #BSUB -o {cond_fold}/trainNN.out\n\
            #BSUB -e {cond_fold}/trainNN.err\n\
            #BSUB -a 'gpu openmp'\n\
            #BSUB -R pascal\n\
            #BSUB -P phys3b\n\
            source /home/phys3b/Envs/keras_tf/bin/activate\n\
            nvidia-smi\n\
            export CUDA_VISIBLE_DEVICES=`/home/phys3b/etc/check_gpu.py 2`\n\
            if [ '$CUDA_VISIBLE_DEVICES' = '-1' ];\n\
            then\
                echo '##### GPUs busy. Restart job later.' exit 1\
            else\
                echo 'Found free GPU devices :'\n\
                echo 'CUDA_VISIBLE_DEVICES =  $CUDA_VISIBLE_DEVICES'\n\
            fi\n\
            python {NN_recofolder}/Neural_Network.py {args}\n".\
                    format(request_cpus = request_cpus,
                           cond_fold=condor_folder,
                           args=arguments,
                           NN_recofolder=thisfolder,
                           mem_request=request_memory)

    return submit_info

