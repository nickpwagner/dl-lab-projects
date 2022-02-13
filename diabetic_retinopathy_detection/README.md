# IMPORTANT
Our code was developed via Live Share in VS Code, meaning the person that commits is not meaningful.
Both students contributed equally to the code.

# Team08
- David Unger (st172353)
- Nick Wagner (st175644)

# How to run the code
1. Clone the repository "dl-lab-team08" to the GPU server wherever you want.
```sh
git clone https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team08.git
```
2. Navigate into the "diabetic_retinopathy_detection" folder.
3. Download one of the two prepared datasets, unpack and put it in the same folder. 
  - Downscaled original: https://drive.google.com/uc?id=1TjyYDws_yFafhVtq2r5itv9SU1EnmXjg
  - Downscaled graham: https://drive.google.com/uc?id=1l5TIieAtew7n7BMlzLLISRpvX2WL5nFK
6. Run the code with "python3 main.py --train=True --epochs=20 --log_wandb="offline" --p="path_where_you_stored_the_ds"
```py
python3 main.py --train=True --epochs=20 --log_wandb="offline" --p="path_where_you_stored_the_ds"
```

