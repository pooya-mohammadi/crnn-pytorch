# CRNN-Pytorch

Pytorch implementation of CRNN. In this repository we'll explain how to train a license plate-recognition model.

## Installation:
```commandline
pip install -r requirements.txt
```

## Dataset 
Before training the model, it's a good practice to calculate the `mean` and `std` of the input dataset and therefore 
normalize the model using proper values instead of merely normalizing with magical `0.5`. Before diving into the code 
make sure that the dataset has the following characters:
```commandline
├── data-dir
│   ├── train
│   │  ├──<text>_<index_01>.jpg
│   │  ├──<text>_<index_02>.jpg
│   │  ├──...
│   ├── val
│   │  ├──<text>_<index_01>.jpg
│   │  ├──<text>_<index_02>.jpg
...
```
**NOTE:** Only `.jpg`, `.png`, and `.jpeg` extensions are supported!

Then checkout the `alphabets.py` module. It contains the alphabets characters that are required for training. 
If the existing alphabets do not meet your requirements create a new dictionary containing your required alphabets and 
then add it to the `ALPHABETS` variable with a specific name. You can get your character set using the following command:
```commandline
python get_character_sets.py --data_directory <path-to-dataset>
```




# References
1. https://github.com/pooya-mohammadi/deep_utils
2. 