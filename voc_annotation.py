import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from train import seed_everything


# Hierarchical cross-validation divides data into training, validation, and test sets


seed=967
test_percent = 0.2
#-------------------------------------------------------#
# Point to the folder where the VOC dataset is located
# Default pointing to the VOC dataset in the root directory
#-------------------------------------------------------#
VOCdevkit_path = 'VOCdevkit'
def process_fold(fold, list, train_seg,total_seg,train_indices, val_indices, saveBasePath, segfilepath):
    print(f"Fold {fold + 1}/{k_folds}")
    train = [list[i] for i in train_indices]
    val = [list[i] for i in val_indices]
    print("train size", len(train))
    print("val size", len(val))
    ftest = open(os.path.join(saveBasePath, f'test_fold{fold}.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, f'train_fold{fold}.txt'), 'w')
    fval = open(os.path.join(saveBasePath, f'val_fold{fold}.txt'), 'w')
    for i in list:
        name = train_seg[i][:-4] + '\n'
        if i in train:
            ftrain.write(name)
            if int(name)>1000:
                ftrain.write(name)
        elif i in val:
            fval.write(name)
            if int(name)>1000:
                fval.write(name)
    for n in total_seg:
        if n not in train_seg:
            ftest.write(n[:-4] + '\n')
    ftrain.close()
    fval.close()
    ftest.close()
    print("Check datasets format for this fold.")
    classes_nums = np.zeros([256], int)
    for i in tqdm(list):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError(f"Tagged image {name} was not detected, please check if the file exists in the specific path and if the suffix is png.")
        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print(f"The label image {name} has a shape of {str(np.shape(png))} and is not a grayscale or octet color image, please double check the dataset format.")
            print("The label image needs to be either grayscale or eight-bit color, and the value of each pixel point of the label is the category to which the pixel point belongs.")
        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
    print("Prints the value and number of pixel points.")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("It was detected that the value of the pixel point in the label contains only 0 and 255, and the data format is incorrect.")
        print("The binary classification problem requires the labels to be modified to have a pixel point value of 0 for the background and a pixel point value of 1 for the target.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("Detected that the label contains only background pixel points and that there is an error in the data format, please double check the dataset format.")
    print("Images in JPEGImages should be .jpg files, and images in SegmentationClass should be .png files.")

if __name__ == "__main__":
    seed_everything(seed)
    print("Generate txt in ImageSets.")
    segfilepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    temp_seg = os.listdir(segfilepath)
    total_seg = [seg for seg in temp_seg if seg.endswith(".png")]
    total = [seg[: -4] for seg in temp_seg if seg.endswith(".png")]
    T,F=[],[]
    for i in total:
        if int(i)>1000:
            F.append(i)
        else:
            T.append(i)
    train_size_T = int((1 - test_percent) * len(T))
    train_size_F = int((1 - test_percent) * len(F))
    T_train = random.sample(T, train_size_T)
    F_train = random.sample(F, train_size_F)
    train=T_train+F_train
    num = len(train)
    list = np.arange(num)
    y=[]
    for name in train:
        true_label = 0 if int(name) > 1000 else 1
        y.append(true_label)
    for i in range(len(train)):
        train[i]=train[i]+'.png'
    k_folds = 5  # Define the number of folds for cross-validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for fold, (train_index, val_index) in enumerate(skf.split(train, y)):
        process_fold(fold, list, train,total_seg, train_index, val_index, saveBasePath, segfilepath)
        print("test size", len(total)-train_size_T-train_size_F)