import numpy as np

def replace_zero_with_nan(df):
    df=df.copy() 
    zero_as_missing_cols=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for col in zero_as_missing_cols:
        if col in df.columns:
            df[col]=df[col].replace(0,np.nan)
    return df
