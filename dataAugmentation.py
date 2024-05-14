import math
import cv2
import numpy as np
import random
import os
import re

#https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def demo():
    """
    Demos the largest_rotated_rect function
    """

    image = cv2.imread("etiketten/2.png")
    image_height, image_width = image.shape[0:2]

    cv2.imshow("Original Image", image)

    print('Press [enter] to begin the demo')
    print ('Press [q] or Escape to quit')

    key = cv2.waitKey(0)
    if key == ord("q") or key == 27:
        exit()

    for i in np.arange(0, 360, 0.5):
        image_orig = np.copy(image)
        image_rotated = rotate_image(image, i)
        image_rotated_cropped = crop_around_center(
            image_rotated,
            *largest_rotated_rect(
                image_width,
                image_height,
                math.radians(i)
            )
        )

        key = cv2.waitKey(2)
        if(key == ord("q") or key == 27):
            exit()

        cv2.imshow("Original Image", image_orig)
        cv2.imshow("Rotated Image", image_rotated)
        cv2.imshow("Cropped Image", image_rotated_cropped)


def horizontal_flip(image, x, width):
   """
   Flips image horizontally and changes bounding box accordingly.

   Keyword arguments:
   image -- openCV image object
   x -- the x-coordinate of the bbox
   width -- the width of the bbox

   Return values:
   flipped_image -- the flipped image
   new_x -- new x-ccordinate of the bbox
   """
   flipped_image = cv2.flip(image, 1)
   image_width = image.shape[1]
   new_x = image_width - (x + width)
   return flipped_image, new_x


def random_lighting(image, brightness_range=(-10, 50), contrast_range=(0.2, 1.5)):
   """
   Changes brightness and contrast of an image within certain ranges.

   Keyword arguments:
   image -- openCV image object
   brigthness_range -- range of random brightness level (DEFAULT -30, 80)
   contrast_range -- range of random contrast level (DEFAULT 0.2, 1.5)

   Return variables:
   image -- openCV image object with adjusted brightness and contrast
   """
   image = image.astype(np.float32)
   brightness = np.random.uniform(brightness_range[0], brightness_range[1])
   image += brightness
   contrast = np.random.uniform(contrast_range[0], contrast_range[1])
   image *= contrast
   image = np.clip(image, 0, 255)
   image = image.astype(np.uint8)
   return image


def save_image(image, output_dir, image_name, append):
   """
   Function to make up new image name and save image to given folder.

   Keyword arguments:
   image -- openCV image object
   output_dir -- path of output folder
   image_name -- original image name
   append -- appendix for new image name

   Return variables:
   final_new_path -- path of the image
   """
   os.makedirs(output_dir, exist_ok=True)
   removed_ending = re.sub(r'\.jpe?g$', '', image_name, flags=re.IGNORECASE)
   removed_old_path = re.sub(r'positive_resized/', '', removed_ending)
   new_name = removed_old_path + append + '.jpg'
   final_new_path = os.path.join(output_dir, new_name)
   cv2.imwrite(final_new_path, image)
   return final_new_path


def transform_images(annotation_file, transforms_file):
    """
    Function to perform transformations to already annotated images.

    Keyword arguments:
    annotation_file -- .txt file of annotations
    transforms_file -- .txt file where transformed image paths and bboxes get written to

    Return values:
    none
    """
    with open(annotation_file, 'r') as file:
        lines = file.readlines()

    with open(transforms_file, 'w') as output:
        for line in lines:
            values = line.strip().split()
            image_path = values[0]
            x = values[2]
            y = values[3]
            width = values[4]
            height = values[5]
            x = int(x)
            width = int(width)
            image = cv2.imread(image_path)

            rotated_image = rotate_image(image, random.uniform(5, 15))
            lit_image = random_lighting(image)
            
            normal_path = save_image(image, 'transformed_images', image_path, 'normal')
            rotated_path = save_image(rotated_image, 'transformed_images', image_path, 'rotated')
            lit_path = save_image(lit_image, 'transformed_images', image_path, 'lit')

            output.write(f"{normal_path} 1 {x} {y} {width} {height}\n")
            output.write(f"{rotated_path} 1 {x} {y} {width} {height}\n")
            output.write(f"{lit_path} 1 {x} {y} {width} {height}\n")


transform_images('training/train/train_pos.txt', 'transformations.txt')



