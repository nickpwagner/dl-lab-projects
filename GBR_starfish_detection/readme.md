## Project 2 - Protect the Great Barrier Reef

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
2. Navigate into the "GBR_starfish_detection" folder.
3. Download the following dataset with downscaled images. (3GB) 
   https://drive.google.com/uc?id=1pJM8yoVMKrptXCZcLT26J5R7MFLHIzo9
6. Run the code with the following command (remeber to put in the path):
```py
python3 main.py -p="path_where_you_stored_the_ds" --wandb_model={"New"} --dataset_slice_end=23500
```

