import os
from pathlib import Path
import yaml
import logging
from joblib import dump

from src.training.config.settings import Settings

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit ,StratifiedGroupKFold, RandomizedSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  classification_report
from dotenv import load_dotenv
load_dotenv()

def training_heart_disease_pred_model():
    try:
        settings=Settings()
        LOG_PATH=Path(settings.log_path)
        DATASET_PATH=Path(settings.heart_disease_dataset_path)
        MODEL_PATH=Path(settings.heart_disease_model_path)
        HEART_DISEASE_PARAMS_YAML_PATH=Path(settings.heart_disease_params_yaml_path)

        TARGET_COL=settings.heart_disease_target_col
        TEST_SIZE=settings.test_size
        RANDOM_STATE=settings.random_state

        LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
        MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | heart | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(filename=LOG_PATH)
            ]
        )

        df=pd.read_csv(DATASET_PATH)
        logging.info(f"DataSet loaded with Shape:{df.shape}")
        X=df.drop(columns=TARGET_COL)
        y=df[TARGET_COL]

        row_signature=pd.util.hash_pandas_object(X, index=False)
        gss=GroupShuffleSplit(n_splits=5, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_index, test_index = next(gss.split(X, y, groups=row_signature))
        X_train , y_train = X.iloc[train_index] , y.iloc[train_index]
        X_test , y_test = X.iloc[test_index] , y.iloc[test_index]
        logging.info(f"X_train.shape:{X_train.shape}  |  X_test.shape:{X_test.shape}")
        
        with open(HEART_DISEASE_PARAMS_YAML_PATH,"r") as file:
            params=yaml.safe_load(file)

        pipeline=Pipeline(steps=[
            ("scaler",StandardScaler()),
            ("model",RandomForestClassifier(random_state=RANDOM_STATE,**params))
        ])

        pipeline.fit(X_train,y_train)
        logging.info("Model Training completed")

        y_train_pred=pipeline.predict(X_train)
        y_test_pred=pipeline.predict(X_test)
        logging.info(f"Train DataSet Pred Metrics:\n{classification_report(y_train,y_train_pred)}")
        logging.info(f"Test  DataSet Pred Metrics:\n{classification_report(y_test ,y_test_pred )}")

        dump(pipeline,filename=MODEL_PATH)
        logging.info(f"Trained Model saved at:\n{MODEL_PATH}")
        logging.info(f"Training Script Completed")  


    except Exception as e:
        logging.info("Error in training_heart_disease_pred_model",e)
        raise

if __name__=="__main__" :
    training_heart_disease_pred_model()

