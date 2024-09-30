import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def display_label_variations():
    img1 = cv.imread('Auspragungen/1.png')
    img2 = cv.imread('Auspragungen/2.png')
    img3 = cv.imread('Auspragungen/9.png')
    img4 = cv.imread('Auspragungen/8.png')
    img5 = cv.imread('Auspragungen/4.png')
    img6 = cv.imread('Auspragungen/3.png')

    _, axs = plt.subplots(2, 3, figsize=(12, 8))

    axs[0, 0].imshow(img1)
    axs[0, 1].imshow(img2)
    axs[0, 2].imshow(img3)
    axs[1, 0].imshow(img4)
    axs[1, 1].imshow(img5)
    axs[1, 2].imshow(img6)

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.suptitle('Label Variations')

    plt.show()


def display_data_transformations():
    img1 = cv.imread('positiveAugmentation/blur.jpg')
    img2 = cv.imread('positiveAugmentation/contrast.jpg')
    img3 = cv.imread('positiveAugmentation/lit.jpg')
    img4 = cv.imread('positiveAugmentation/noise.jpg')
    img5 = cv.imread('positiveAugmentation/spnoise.jpg')
    img6 = cv.imread('positiveAugmentation/rotated.jpg')

    _, axs = plt.subplots(2, 3, figsize=(12, 8))

    axs[0, 0].imshow(img1)
    axs[0, 1].imshow(img2)
    axs[0, 2].imshow(img3)
    axs[1, 0].imshow(img4)
    axs[1, 1].imshow(img5)
    axs[1, 2].imshow(img6)

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.suptitle('Data Augmentation')

    plt.show()


def show_train_test_split():
    """
    This function shows the distribution of positive and negative data as well as the train test split.

    Parameters:
    none

    Returns:
    none
    """
    total_train = 1627 + 791
    total_test = 407 + 199

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.bar(0, 1627, color='#0066cc', width = 0.3, label='training')
    ax.bar(0, 407, color='#99ccff', width = 0.3, bottom=1627, label='test')

    ax.bar(1, 791, color='#0066cc', width = 0.3, label='training')
    ax.bar(1, 199, color='#99ccff', width = 0.3, bottom=791, label='test')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['negative', 'positive'])

    ax.set_xlabel('Data type', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Train-test-split of positive and negative data', fontweight='bold')

    ax.legend(['training', 'test'], loc='upper right')

    plt.subplots_adjust(bottom=0.2)

    plt.show()


def show_bbox_sizes():
    widths = []
    heights = []
    with open('training/pos.txt', 'r') as file:
        for line in file:
            values = line.strip().split()
            widths.append(int(values[4]))
            heights.append(int(values[5]))

    areas = [width * height for width, height in zip(widths, heights)]

    # Create the histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.hist(widths, bins=30, alpha=0.5, label='Width')
    #ax.hist(heights, bins=30, alpha=0.5, label='Height')
    ax.hist(areas, bins=20, alpha=0.5, label='Area')
    ax.set_xlabel('Bounding Box Dimensions')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Bounding Box Sizes')
    ax.legend()
    plt.show()

    print(np.mean(widths))
    print(np.mean(heights))

#show_train_test_split()