from settings import train, validate, models
from utils import read_files, create_folders
from model import train_model, validate_model

if __name__ == "__main__":

    print('Please cite the following paper when using the POPF predictor: Ingwersen EW, Bereska JI, Balduzzi A, Janssen BV, Besselink MG, Kazemier G, Marchegiani G, Malleo G, Marquering HA, Nio CY, de Robertis R, Salvia R, Steyerberg EW, Stoker J, Struik F, Verpalen IM, Daams F; Pancreatobiliary and Hepatic Artificial Intelligence Research (PHAIR) consortium. Radiomics preoperative-Fistula Risk Score (RAD-FRS) for pancreatoduodenectomy: development and external validation. BJS Open. 2023 Sep 5;7(5):zrad100. doi: 10.1093/bjsopen/zrad100. PMID: 37811791.')

    df = read_files()
    create_folders()

    if train: train_model(df, models)
    if validate: validate_model(df, models)
