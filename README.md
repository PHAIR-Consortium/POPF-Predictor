# POPF-Predictor
Binary classifier for predicting POPF - code PLUS trained and externally validated model 


# Installation 


The POPF predictor has been tested on MacOS (Monterey, Version 12.6) and Windows 11. We do not provide support for other operating systems.

The POPF predictor does not requires a GPU. For training, we recommend a strong CPU. At least 6 CPU cores (tested on 6-Core Intel Core i5).

We very strongly recommend you install the POPF predictor in a virtual environment. 

Python 2 is deprecated and not supported. Please make sure you are using Python 3.

For more information about the POPF predictor, please read the following paper:

TODO: add citation here

Please also cite this paper if you are using the POPF predictor for your research!

Follow these steps to run the POPF predictor:

1. Install the POPF predictor
```
 git clone https://github.com/PHAIR-Amsterdam/POPF-Predictor.git
 cd POPF-Predictor
 pip install -e .
```
2. The POPF predictor needs to know where you want to save input data, trained models, and results. For this you need to set a path variable in ```settings.py``` and save your data in the correct folder structure:

    2.1 Save radiomics file (xlsx) and key file (xlsx) in ```data``` folder. The key file has the name ```db_basis_YOURMODELNAME.xlsx``` and contains the patient identifiers (column PP) and the events (column Events). The radiomics file has the name ```YOURMODELNAME_radiomics.xlsx``` and contains Pyradiomics radiomic features and patient identifiers (column PP). 
    
    2.2 Set the model name under ```file_extension```. The model name must contain either ```training``` or ```validate```, depending on your use case. For example, if you set ```file_extension``` to ```POPF_training```, your key file must be saved as ```data/db_basis_POPF_training.xlsx``` and your radiomics file must be saved as ```data/POPF_training_radiomics.xlsx```. Trained models will be saved in the folder ```POPF_training_models``` and results will be saved under ```POPF_training_results```.

3. The POPF predictor needs to know which settings to follow for training. For this you need to specify several variables in ```settings.py```:
 
  3.1 Select the use of preoprocessing methods over- and undersampling (```oversample``` and ```undersample```), noise (```noise```), lasso feature reduction (```lasso```). 
  
  3.2 Select which models you would like to train under ```models``` (choice of support vector machine ```svm```, logistic regression ```lr```, k-nearest neighbor ```knn```, and random forest ```rf```).
  
  3.3 Specify how many times you wish to train each model under ```num_loops```. 
  
