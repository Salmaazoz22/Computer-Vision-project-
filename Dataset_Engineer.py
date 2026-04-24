# import zipfile

# zip_path = r"D:\Python\CVproject\archive.zip"
# extract_path = r"D:\Python\CVproject\GTSRB_data"

# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_path)

# print("Extraction completed!")



#HERE  to check number of classes should be 43  if less than 43 then there is a problem with the dataset and we need to re-download it and extract it again.
#or we are working on a subset of the dataset which is not recommended for training a model.


import os
data_path = "D:\Python\CVproject\GTSRB\\Train"

classes = os.listdir(data_path)
print("Number of classes:", len(classes))

#HERE  to know the number of images in each class to check if there are any classes with very few images which might cause problems during training 
# and we can decide to either remove those classes or augment the data for those classes to balance the dataset.
import pandas as pd

data = []

for cls in classes:
    path = os.path.join(data_path, cls)
    count = len(os.listdir(path))
    data.append((cls, count))

df = pd.DataFrame(data, columns=["class", "count"])
print(df)

#HERE to visualize the distribution of images across classes to identify any imbalances in the dataset 
# which can affect the performance of the model and we can decide to either remove some classes or augment the data for underrepresented classes to balance the dataset.

import matplotlib.pyplot as plt

plt.bar(df["class"], df["count"])
plt.xticks(rotation=90)
plt.title("Images per Class")
plt.show()


#HERE to visualize some sample images from the dataset to get a better understanding of the data and to check if there are any issues with the images 
# such as low resolution, blurriness, or incorrect labels which can affect the performance of the model and we can decide to either remove those images or correct the labels to improve the quality of the dataset.
import cv2
import matplotlib.pyplot as plt

for cls in classes[:5]:
    img_path = os.path.join(data_path, cls, os.listdir(os.path.join(data_path, cls))[0])
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(f"Class {cls}")
    plt.axis("off")
    plt.show()

#we check first the lighting .. if the images are too dark or too bright or there is adirect sunlight in the images which means the module need to learn how to deal with variations
#then we check the size of the images .. if the images are too small or too large which means we need to resize them to a standard size for training the model
#The variation in the images .. if there are variations in the images such as different angles, backgrounds, or occlusions which means the model need to learn how to deal with these variations 


#HERE to remove any corrupted or duplicate images from the dataset , one image byza can break the training or cause sudden errors 

from PIL import Image
import os

def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()  # يتأكد إنها صورة سليمة
        return True
    except:
        return False


data_path = "GTSRB/Train"

for root, dirs, files in os.walk(data_path):
    for file in files:
        path = os.path.join(root, file)

        if not is_valid_image(path):
            print("Removing:", path)
            os.remove(path)


#HERE to remove any duplicate images from the dataset which can cause overfitting and reduce the generalization of the model and we can use hashing to identify duplicate images and remove them from the dataset.
#Duplicates make the model memorize the training data instead of learning to generalize from it, which can lead to poor performance on unseen data.

import hashlib

def file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

seen = set()

for root, dirs, files in os.walk(data_path):
    for file in files:
        path = os.path.join(root, file)
        h = file_hash(path)

        if h in seen:
            print("Duplicate removed:", path)
            os.remove(path)
        else:
            seen.add(h)
#HERE to check if there are any empty classes in the dataset The folder name = label of the class and if there are any empty folders it means there are no images for that class which can cause problems during training 
for cls in os.listdir(data_path):
    path = os.path.join(data_path, cls)
    
    if len(os.listdir(path)) == 0:
        print("Empty class:", cls)


#HERE to check if there are any unreadable images in the dataset which can cause problems during training and we can use OpenCV to try to read each image and if it fails we can remove that image from the dataset.
import cv2

for root, dirs, files in os.walk(data_path):
    for file in files:
        path = os.path.join(root, file)

        img = cv2.imread(path)
        if img is None:
            print("Unreadable:", path)
            os.remove(path)


#as we have unbalanced dataset we need to create a balanced subset 
#the main idea is we choose a fixed number of images from each class to create a balanced dataset for training the model and we can use random sampling to select images from each class and create a new dataset with an equal number of images for each class.


import os
import random
import shutil

source = "GTSRB/Train"
dest = "balanced_dataset"

TARGET = 200

for cls in os.listdir(source):
    src_folder = os.path.join(source, cls)
    images = os.listdir(src_folder)

    if len(images) >= TARGET:
        selected = random.sample(images, TARGET)
    else:
        
        selected = images

    dst_folder = os.path.join(dest, cls)
    os.makedirs(dst_folder, exist_ok=True)

    for img in selected:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(dst_folder, img)
        )
#HERE to check the number of images in each class in the new subset dataset to ensure that we have a balanced dataset for training the model and we can use os.listdir to count the number of images in each class and print the results to verify that we have the same number of images for each class.
for cls in os.listdir(dest):
    print(cls, "->", len(os.listdir(os.path.join(dest, cls))))


#Finally, we need to split the balanced dataset into training, validation, and test sets to evaluate the performance of the model on unseen data and we can use random sampling to split the dataset into different sets and create separate folders for each set to organize the data for training and evaluation.
# CREATE THE MAIN FOLDER 
import os

base = "dataset"

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(base, split), exist_ok=True)

#CREATE CLASSES IN EVERY SPLIT FOLDER
source = "balanced_dataset"

classes = os.listdir(source)

for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(base, split, cls), exist_ok=True)

#THIS split helps to separate the data into training, validation, and test sets 


#HERE Split dataset into 70 train, 15 validation, and 15 test they are totally separated 
#can't split the data all at once , we need to split the data for each class separately to ensure that we have a balanced distribution of images across the different sets 
import os
import random
import shutil

source = "balanced_dataset"
dest = "dataset"

classes = os.listdir(source)

for cls in classes:
    images = os.listdir(os.path.join(source, cls))
    random.shuffle(images)

    total = len(images)

    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split, split_imgs in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):
        split_path = os.path.join(dest, split, cls)
        os.makedirs(split_path, exist_ok=True)

        for img in split_imgs:
            shutil.copy(
                os.path.join(source, cls, img),
                os.path.join(split_path, img)
            )

#HERE to check the number of images in each class for each split to ensure that we have a balanced distribution of images across the training, validation, and test sets and we can use os.listdir to count the number of images in each class for each split and print the results to verify that we have the expected distribution of images for training and evaluation.
for split in ["train", "val", "test"]:
    print("\n", split.upper())
    
    for cls in sorted(os.listdir(f"dataset/{split}")):
        count = len(os.listdir(f"dataset/{split}/{cls}"))
        print(cls, "->", count)

#in GTSRB we are working classification only , every image is a singlr class
#HERE to create a CSV file with the image paths and their corresponding labels to facilitate the training of the model and we can use pandas to create a DataFrame with the image paths and labels and then save it as a CSV file for easy access during training.
import os
import pandas as pd

data = []

base = "dataset"

for split in ["train", "val", "test"]:
    for cls in os.listdir(os.path.join(base, split)):
        folder = os.path.join(base, split, cls)

        for img in os.listdir(folder):
            path = f"{split}/{cls}/{img}"
            data.append([path, cls])

df = pd.DataFrame(data, columns=["image_path", "label"])
df.to_csv("labels.csv", index=False)



#HERE we are going to resize images so every picture has the same size , shape and suitable for the CNN 


import cv2
import os

base = "dataset"
SIZE = 64

for split in ["train", "val", "test"]:
    split_path = os.path.join(base, split)

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)

        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)

            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, (SIZE, SIZE))
                cv2.imwrite(img_path, img)

#HERE we create metadata csv mesh fahmaha 2wy google it 

import os
import pandas as pd

base = "dataset"
data = []

for split in ["train", "val", "test"]:
    split_path = os.path.join(base, split)

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)

        for img in os.listdir(cls_path):
            img_path = f"{split}/{cls}/{img}"
            data.append([img_path, cls])

df = pd.DataFrame(data, columns=["image_path", "label"])

df.to_csv("dataset_metadata.csv", index=False)

print("CSV saved successfully!")



#TO CHECKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK



#To check that there is no Unbalanced classes
import os

dest = "balanced_dataset"
for cls in os.listdir(dest):
    print(cls, "->", len(os.listdir(os.path.join(dest, cls))))



#to check that there is no empty classes 
import os

for split in ["train", "val", "test"]:
    path = os.path.join("dataset", split)
    for cls in os.listdir(path):
        if len(os.listdir(os.path.join(path, cls))) == 0:
            print("Empty class:", cls, "in", split)



#to check that there is no Different image sizes
import cv2
import os

SIZE = (64, 64)
base = "dataset"

for split in ["train", "val", "test"]:
    for cls in os.listdir(os.path.join(base, split)):
        folder = os.path.join(base, split, cls)
        for img_name in os.listdir(folder)[:5]:  # جربي أول 5 صور بس
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is not None and img.shape[:2] != SIZE:
                print("Wrong size:", img_path, img.shape)



#to check that there is no Unreadable images in the dataset
import os

val_path = "dataset/val"
for cls in os.listdir(val_path):
    print(cls, "->", len(os.listdir(os.path.join(val_path, cls))))
