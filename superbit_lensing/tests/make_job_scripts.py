import glob
import argparse
import io
import os
import pdb
import re

parser = argparse.ArgumentParser()

parser.add_argument('run_name',type=str,
                    help='Root name of simulations to process')
parser.add_argument('--js_dir',type=str, default=None,
                    help='Output directory for job scripts')
parser.add_argument('--base_dir',type=str, default=None,
                    help='Directory containing star catalogs & images')



def print_bashfile(cluster_name, config_name, realization):

    output = io.StringIO()

    print('#!/bin/sh', file=output)
    print('#SBATCH -t 4:30:00', file=output)
    print('#SBATCH -N 1', file=output) 
    print('#SBATCH -n 18', file=output)
    print('#SBATCH --mem-per-cpu=5g', file=output)
    print(f'#SBATCH -J {cluster_name}_{realization}', file=output)
    print('#SBATCH -v', file=output) 
    print('#SBATCH --mail-type=ALL', file=output)
    print('#SBATCH --mail-user=jmac.ftw@gmail.com', file=output)
    print(f'#SBATCH -o {cluster_name}_{realization}.out', file=output)
    print(f'#SBATCH -e {cluster_name}_{realization}.err', file=output)

    print('', file=output)
    print('', file=output)
        
    print('dirname="slurm_outfiles"', file=output)
    print('if [ ! -d "$dirname" ]', file=output)
    print('then', file=output)
    print('     echo " Directory $dirname does not exist. Creating now"', file=output)
    print('     mkdir -p -- "$dirname"', file=output)
    print('     echo " $dirname created"', file=output)
    print(' else', file=output)
    print('     echo " Directory $dirname exists"', file=output)
    print(' fi', file=output)
    print('', file=output)
    print(' echo "Proceeding with code..."', file=output)

    print('', file=output)
    print('', file=output)

    print(f'python /users/jmcclear/data/superbit/superbit-metacal/superbit_lensing/run_pipe.py {config_name}',file=output)
    print('', file=output)
    print('', file=output)

    print(f'mv {cluster_name}_{realization}.out {cluster_name}_{realization}.err "$dirname"', file=output)
    
    return output.getvalue()

   


def main():

    args = parser.parse_args()
    run_name = args.run_name
    js_dir = args.js_dir
    base_dir = args.base_dir
    
    if js_dir == None:
        js_dir = '/gpfs/data/idellant/jmcclear/superbit/superbit-metacal/job_scripts'
    if base_dir == None:
        base_dir = '/gpfs/scratch/jmcclear/mock-data-forecasting/'


    # Make sure that the run_dir and job_dir exist
    run_dir = os.path.join(base_dir, run_name)
    
    if not os.path.isdir(run_dir):
        raise OSError(f'run_dir {run_dir} not found! Check supplied base_dir and run_name')

    if not os.path.exists(os.path.join(js_dir,run_name)):
        os.mkdir(os.path.join(js_dir,run_name))

    # For convenience, write sbatch commands to a text file 
    sb = open('sbatch_submit_jobs.txt','w')

    # get cluster names and loop over them 
    clusters_glob = glob.glob(f'{run_dir}/*')

    for cluster in clusters_glob:
        
        cluster_name = cluster.replace(f'{run_dir}/','')
        sb.write(f'##\n## {cluster_name}\n##\n\n')

        configs = glob.glob(f'{run_dir}/{cluster_name}/r*/*{cluster_name}*yaml')
        configs.sort(key=lambda x: int(re.search('r[0-9]{1,2}', x.split(f'{run_name}')[1]).group()[1:]))
        #pdb.set_trace()
        
        for i,config in enumerate(configs):

            realization = re.search('r[0-9]{1,2}',config).group()
            bash_name = os.path.join(js_dir,run_name,f'job_{cluster_name}_{realization}.sh')

            with open(bash_name, 'w') as f:
                
                script = print_bashfile(cluster_name, config, realization)
                f.write(script)
                f.close()

            sb.write(f'sbatch {bash_name}\n')

            if ((i+1) % 10 == 0):
                sb.write('\n')

        
    sb.close()

    print("Done!")
    
        
    
    
if __name__ == main():

    import pdb, traceback, sys
    
    try:
        main()
        
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

