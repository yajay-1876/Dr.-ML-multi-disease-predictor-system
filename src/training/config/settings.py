from pydantic_settings import BaseSettings

# .env is searched relative to the CURRENT WORKING DIRECTORY (where you run Python from), not necessarily where settings.py exists.

class Settings(BaseSettings):

    log_path: str
    diabetes_dataset_path: str
    heart_disease_dataset_path: str
    diabetes_model_path: str
    heart_disease_model_path: str
    diabetes_target_col: str
    heart_disease_target_col: str
    diabetes_params_yaml_path: str
    heart_disease_params_yaml_path: str
    test_size: float
    random_state: int
    
    class Config:
        env_file= ".env"
        env_file_encoding= "utf-8"
        extra= "allow"
