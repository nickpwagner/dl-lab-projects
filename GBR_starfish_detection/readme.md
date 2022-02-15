## Project 2 - Protect the Great Barrier Reef
<img src="https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team08/blob/master/GBR_starfish_detection/220208_GBR_Train_Video.gif?raw=true">

## IMPORTANT
Our code was developed via Live Share in VS Code, meaning the person that commits is not meaningful.
Both students contributed equally to the code.
## Team08
- David Unger (st172353)
- Nick Wagner (st175644)

# How to run the code
## Option 1: Simply go to the following folder and run the GBR.sh script with sbatch.
```/misc/home/RUS_CIP/st175644```<br />
<img width="135" alt="GPU_server" src="https://media.github.tik.uni-stuttgart.de/user/3602/files/22888200-8e42-11ec-87f9-53f9abc07093">

## Option 2: Set the environment up yourself.
### Clone the repository "dl-lab-team08" to the GPU server wherever you want.
```sh
git clone https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team08.git
```
### Download the following dataset with downscaled images. (3GB) 
   https://drive.google.com/uc?id=1pJM8yoVMKrptXCZcLT26J5R7MFLHIzo9

### Two alternatives for the shell script:<br />
EITHER use the existing shell script that can be found here, update the dataset path in the last line marked with "-p" and run the sbatch command with it: 
```sh
/misc/home/RUS_CIP/st175644/GBR.sh
```
OR create your own shell script and change the dataset path in the last line marked with "-p":
 ```sh
#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=T8_GBR
#SBATCH --output=job_name-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# Activate everything you need
module load cuda/11.2
wandb login 37262d20054e8dbf092705158103cd02e31691d6
# Run your python code
cd dl-lab-21w-team08/GBR_starfish_detection
python3 main.py --wandb_model="New" --dataset_slice_end=23500 -p="/misc/home/RUS_CIP/st175644/GBR_dataset/"
```

## If anything is unclear, feel free to contact us on Slack. Enjoy :-)
