import numpy as np
import random
import os

#Splitting the data into train and test while also keeping data in positive negative split

#random.seed(42)

positive_file = 'pos.txt'
negative_file = 'neg.txt'

train_dir = 'train'
test_dir = 'test'

train_percentage = 0.8

with open(positive_file, 'r') as file:
   positive_files = file.read().splitlines()

with open(negative_file, 'r') as file:
   negative_files = file.read().splitlines()


random.shuffle(positive_files)
random.shuffle(negative_files)


train_samples_pos = int(train_percentage * len(positive_files))

train_samples_neg = int(train_percentage * len(negative_files))


with open(os.path.join(train_dir, 'train_pos.txt'), 'w') as file:
   for file_name in positive_files[:train_samples_pos]:
       file.write(file_name + '\n')

with open(os.path.join(test_dir, 'test_pos.txt'), 'w') as file:
   for file_name in positive_files[train_samples_pos:]:
       file.write(file_name + '\n')

with open(os.path.join(train_dir, 'train_neg.txt'), 'w') as file:
   for file_name in negative_files[:train_samples_neg]:
       file.write(file_name + '\n')

with open(os.path.join(test_dir, 'test_neg.txt'), 'w') as file:
   for file_name in negative_files[train_samples_neg:]:
       file.write(file_name + '\n')

