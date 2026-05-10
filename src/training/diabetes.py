import pandas as pd
import numpy as np

import logging
import yaml
import os
from pathlib import Path
from joblib import dump

from src.training.config.settings import Settings
from src.common.preprocessing_util import replace_zero_with_nan

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
from dotenv import load_dotenv
load_dotenv()

def train_diabetes_model():
    try:
        # before creating a Settings class we need to run  .env file 
        # This run is to store variables inside .env file to current bash terminal temporarily

        settings=Settings()

        DATASET_PATH=Path(settings.diabetes_dataset_path)
        MODEL_PATH=Path(settings.diabetes_model_path)
        LOG_PATH=Path(settings.log_path)
        DIABETES_PARAMS_YAML_PATH=Path(settings.diabetes_params_yaml_path)

        TARGET_COL=settings.diabetes_target_col
        TEST_SIZE=settings.test_size
        RANDOM_STATE=settings.random_state

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | diabetes | %(message)s",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(filename=LOG_PATH)
            ]
        )

        logging.info("started diabetes model training")
        
        df=pd.read_csv(DATASET_PATH)
        logging.info(f"loaded dataset with shape: {df.shape}")
        X=df.drop(columns=TARGET_COL)
        y=df[TARGET_COL]
        X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

        logging.info(f"X_Train.shape: {X_train.shape}   |   y_train.shape:  {y_train.shape}")

        numerical_features= X_train.select_dtypes(include=[np.number]).columns.to_list()

        numerical_transformer=Pipeline(steps=[
            ("zero_to_nan", FunctionTransformer(replace_zero_with_nan)),
            ("Imputer",SimpleImputer(strategy='median')),
            ("scaler",StandardScaler())    
        ])

        preprocess=ColumnTransformer(transformers=[
            ("num_transformer",numerical_transformer,numerical_features)
        ])

        with open(DIABETES_PARAMS_YAML_PATH, 'r') as file:
            hyperparams=yaml.safe_load(file)
        
        model_params=hyperparams
        model=SVC(random_state=RANDOM_STATE, class_weight='balanced', **model_params)

        pipeline=Pipeline(steps=[
            ("preprocess",preprocess),
            ("model",model)
        ])

        pipeline.fit(X_train,y_train)
        logging.info("Model Training completed")
        
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        logging.info("Train Classification Report:\n" + classification_report(y_train, y_train_pred))
        logging.info("Test Classification Report:\n" + classification_report(y_test, y_test_pred))

        dump(pipeline,MODEL_PATH)
        logging.info(f"Model saved at:\n{MODEL_PATH}")
        logging.info(f"Model Training completed")

    except Exception as e:
        logging.info("Error occured , exiting training script",e)
        raise

if __name__== "__main__":
    train_diabetes_model()
