# CRNN-Pytorch

Pytorch implementation of the CRNN model.
In this repository I explain how to train a license plate-recognition model with pytorch-lightning.

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
then add it to the `ALPHABETS` variable with a specific name. You can get your character set using the following
command:

```commandline
python get_character_sets.py --data_directory <path-to-dataset>
```
The output will be like the following: 
```commandline
[INFO] characters: +ابتثجدزسشصطعقلمنهوپگی۰۱۲۳۴۵۶۷۸۹
```

Finally, run the following command to get the `mean`, `std` and `n_classes` of your input dataset:

```commandline
python dataset.py --train_directory <your-train-dir> --alphabet_name FA_LPR --batch_size 128
```

The output would be like the below:

```commandline
[INFO] MEAN: [0.4845], STD: [0.1884]
[INFO] N_CLASSES: 35 ---> ابپتشثجدزسصطعفقکگلمنوهی+۰۱۲۳۴۵۶۷۸۹
```

Get the stats and replace them with values provided for `MEAN`, `STD`, and `N_CLASSES` in the `settings.py` module under
the `BasicConfig` class.

## Train:
After modifying the aforementioned configs, run the following command to train the model:
```commandline
python train.py
```

## Inference
For inference run the following code:
```commandline
python inference.py --model_path logs/best_model.ckpt --img_path sample_images/۱۴ق۹۱۸۱۱_7073.jpg
```
The output should be like the following:
```commandline
۱۴ق۹۱۸۱۱
```

Image examples:

![](assets/sample_01.png)

![](assets/sample_02.png)

### Sample Persian Dataset is avalable by Amirkabir University of Technology in the following link:
https://ceit.aut.ac.ir/~keyvanrad/download/ML971/project/

Password: ML971Data

### Foot-Notes:
1. For labeling tool checkout my project:
   1. https://github.com/pooya-mohammadi/ocr-labeling-tool
   

# References

1. https://github.com/pooya-mohammadi/deep_utils
2. https://github.com/AryanShekarlaban/
