# Specify name for folder structure
file_extension = 'manual_validate'

# Specify data augmentation techniques (1 = yes, 0 = no)
undersample = 1
oversample = 1
noise = 1
synthetic = 0
lasso = 1

# Specify models to train / validate
models = ['svm', 'lr', 'knn', 'rf']

# Specify number of models you would like to train
num_loops = 25

# Specify test set size for train_test_split
test_set_size = 0.1

if 'validate' in file_extension: validate, train = 1, 0
if 'validate' not in file_extension: validate, train = 0, 1
