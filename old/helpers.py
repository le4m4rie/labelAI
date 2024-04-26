import os
import cv2
import shutil

def generate_negative_description_file():
    """
    Creates a neg.txt file from the negative images folder.
    """
    with open('neg.txt', 'w') as f:
        for filename in os.listdir('negative'):
            f.write('negative/' + filename + '\n')


def resize_to_width(new_width: int, file: str):
   """
   Scales an image to a predefined width.
   """
   img = cv2.imread(file)
   h, w = img.shape[:2]
   ratio = new_width / float(w)
   new_height = int(h * ratio)
   resized_img = cv2.resize(img, (new_width, new_height))
   return resized_img


def resize_images_to_folder():
    """
    Scales the positive images to width 500 and writes them to a new folder
    """
    input_folder = 'positive'

    output_folder = 'positive_resized'

    new_width = 500

    for filename in os.listdir(input_folder):
        img = cv2.imread(os.path.join(input_folder, filename))
        h, w = img.shape[:2]
        ratio = new_width / float(w)
        new_height = int(h * ratio)
        resized_img = cv2.resize(img, (new_width, new_height))

        cv2.imwrite(os.path.join(output_folder, filename), resized_img)


def rename_images():
    folder_path = 'Fotos_Etiketten_Lager'
    new_name = 'IMG'
    count = 414

    for filename in os.listdir(folder_path):
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, '{}_{:04d}.JPG'.format(new_name, count))
        shutil.move(src, dst)
        count +=1
