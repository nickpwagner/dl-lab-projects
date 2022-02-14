## Project 1 - Diabetic Retinopathy Detection

# IMPORTANT
Our code was developed via Live Share in VS Code, meaning the person that commits is not meaningful.
Both students contributed equally to the code.

# Team08
- David Unger (st172353)
- Nick Wagner (st175644)

# How to run the code
## Clone the repository "dl-lab-team08" to the GPU server wherever you want.
```sh
git clone https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team08.git
```
## Download one of the two prepared datasets, unpack and copy it into the same folder as "dl-lab-team08" is. 
  - Downscaled original: https://drive.google.com/uc?id=1TjyYDws_yFafhVtq2r5itv9SU1EnmXjg
  - Downscaled graham: https://drive.google.com/uc?id=1l5TIieAtew7n7BMlzLLISRpvX2WL5nFK

## Two alternatives to run the code:<br />

EITHER use the shell script that can be found here, update the dataset path and run the sbatch command with it: 
```sh
/misc/home/RUS_CIP/st175644/DRD.sh
```
  OR create your own shell script and change the dataset path in the last line marked with "-p":
 ```sh
#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=T8_DRD
#SBATCH --output=job_name-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# Activate everything you need
module load cuda/11.2
# Run your python code
cd dl-lab-21w-team08/diabetic_retinopathy_detection
python3 main.py --train=True --epochs=20 --log_wandb="offline" -p="/misc/home/RUS_CIP/st175644/IDRID_dataset/"
```

## Evaluation
To evaluate a run, you can pass the w&b run into the evaluate script

```
wandb login 37262d20054e8dbf092705158103cd02e31691d6
python3 evaluation.py --mode="multi_class" --evaluate_run="stuttgartteam8/diabetic_retinopathy/hsiu62mg"

# he resultrs from the paper can be reproduced using the following runs: 
python3 evaluation.py --mode="multi_class" "davidu/diabetic_retinopathy/u5ltosw6" -p="/misc/home/RUS_CIP/st175644/IDRID_dataset/"
python3 evaluation.py --mode="binary_class" "davidu/diabetic_retinopathy/i7144mrm" -p= Path to graham dataset
```

## If anything is unclear, check ```/misc/home/RUS_CIP/st175644```. Enjoy :-)
