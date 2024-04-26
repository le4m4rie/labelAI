import cv2 as cv
import os
import torchvision.transforms as transforms

#Code Repository: https://github.com/thomassabbe/drone-object-detection-using-haar-cascades/blob/main/dronevision_real/src/dronevision_library.py

def create_transformation(path_to_importfolder, path_to_exportfolder, start_number, repeat_variable):
    """
    Main definition to create the transformations and store them.

    Keyword arguments:
    path_to_importfolder -- path to the import folder
    path_to_exportfolder -- path to the export folder
    start_number -- start number for writing images to the export folder
    repeat_variable -- amount of times an augment needs to be created from the same image

    Return variables:
    none
    """
    # Create a collection of transformations. The choices which tranformations are arbitrary.
    my_transforms = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.RandomCrop((500, 500)),
        transforms.ColorJitter(brightness=0.5, hue=0.5, saturation=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomGrayscale(p=0.2),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

    # Create a dataset from an image-folder. This image-folder should contain all the sample-images you want to be augmented.
    val_set = datasets.ImageFolder(root=path_to_importfolder, transform=my_transforms)

    # Set a img_num. Default 0, but can be different if you want to add more images upon existing ones.
    img_num = start_number
    # For loop which will repeat 20 times per image in the dataset, explained above.
    # Ex.: 101 sample pictures will result into 2020 pictures.
    for _ in range(repeat_variable):
        for img, label in val_set:
            save_image(img, path_to_exportfolder + str(img_num) + '.jpg')
            img_num += 1



