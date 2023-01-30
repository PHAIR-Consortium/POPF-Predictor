# POPF Predictor
Postoperative pancreatic fistula (POPF) is a complication that can arise following a pancreatoduodenectomy (PD), that negatively impacts short-term and long-term outcomes and cancer recurrence rates. Accurate risk stratification of POPF in the preoperative setting can assist in determining the best surgical approach for high-risk or frail patients. In the perioperative period, high-risk patients for POPF may be candidates for prophylactic treatment with somatostatin analogs. In patients with cystic lesions of the pancreatic head, accurate risk stratification of POPF can help to make the decision of whether to proceed with a PD or consider alternative approaches.4 A recent study demonstrated improved postoperative outcomes for high-risk patients who underwent a total pancreatectomy compared to those who underwent a PD.

Previous research has introduced several POPF prediction models, including the fistula risk score (FRS) by Callery et al. and the updated alternative fistula risk score (ua-FRS) by Mungroop et al.These risk models are commonly used but have limitations, including their reliance on subjective measurements (e.g., intra-operative assessment of the texture of the pancreas) and their inability to provide predictions before surgery.
Radiomics is an approach to extract features from medical images. It enables objective approaches for texture analysis and can uncover new parameters, some invisible to the human eye or mind.  Previous studies have investigated the use of computed tomography (CT)-based radiomics models to predict POPF and showed promising results. 

The POPF Predictor makes the following contributions to the field:
  1. **Standardized baseline:** The POPF predictor is the first standardized POPF prediction benchmark. Without manual effort, researchers can compare their predcition models against the POPF predictor to provide meaningful evidence for proposed improvements.
  2. **Out-of-the-box prediction method:** The POPF predictor is the first plug-and-play tool for POPF prediction. Inexperienced users can use the POPF predictor without need for manual intervention.
  3. **Externally validated:** The POPF predictor is the first publically available **and** externally validated POPF prediction model. The model was developed and internally tested using CT-scans of 118 patients from the Amsterdam University Medical Center. The external test set comprised 57 patients from the Verona University Hospital. The AUROC of the random forest was 0.80 (95% CI: 0.69 – 0.92) in the external test set. The calibration curve indicated that the model's prediction were reliable. The discrimination of the RAD-FRS in the external test set was similar to the FRS (AUC: 0.79) and ua-FRS (AUC: 0.79). 

# Installation and Setup


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

```
POPF-Predictor
├── data
│   └── POPF_training_radiomics.xlsx
│   └── db_basis_POPF_training.xlsx
├── POPF_training_models
│   └── rf_scaler.pkl
│   └── rf_rmodel.pkl
│   └── rf_features.pkl
│   └── rf_x_test.pkl
│   └── rf_y_test.pkl
├── POPF_training_results
│   └── results.csv
│   └── rf_roc_curve.png
│   └── rf_jitter_scores.png
│   └── rf_confusion_matrix.png 
```

3. The POPF predictor needs to know which settings to follow for training. For this you need to specify several variables in ```settings.py```:
 
   3.1 Select the use of preoprocessing methods over- and undersampling (```oversample``` and ```undersample```), noise (```noise```), lasso feature reduction (```lasso```). 
  
   3.2 Select which models you would like to train under ```models``` (choice of support vector machine ```svm```, logistic regression ```lr```, k-nearest neighbor ```knn```, and random forest ```rf```).
  
   3.3 Specify how many times you wish to train each model under ```num_loops```. 
   
   
# Training
 
To train the POPF predictor on your own dataset, follow the instruction under Installation (2) to store your radiomics and key file, set the file extension (must end with ```training```) in ```settings.py```. Finally, run the function ```main.py```.


# Validation
 
To validate the POPF predictor on your own dataset, follow the instruction under Installation (2) to store your radiomics and key file, set the file extension (must end with ```validate```) in ```settings.py```. Finally, run the function ```main.py```.

