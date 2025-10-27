
# preprocess function that preprocesses our data and creates rules for transofrming our test set later on. 

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

def fit_and_preprocess_train(train_df: pd.DataFrame):
    """
    Takes the raw training DataFrame, preprocesses it according to our specific EDA plan,
    and learns the transformation rules.
    
    Args:
        train_df: The raw training pandas DataFrame.
        
    Returns:
        A tuple containing:
        - processed_train_df: The cleaned and preprocessed training DataFrame.
        - transformation_rules: A dictionary containing the learned medians, scaler, and encoder.
    """
    data = train_df.copy()

    # --- Step 1 & 2: Feature Selection and Target Creation ---
    features_to_keep = [
        'phases', 'study_type', 'enrollment_count', 'lead_sponsor_class', 
        'sex', 'minimum_age', 'maximum_age', 'overall_status'
    ]
    data = data[features_to_keep]

    success_stati = ["COMPLETED"]
    failure_stati = ["TERMINATED", "WITHDRAWN", "SUSPENDED"]
    data = data[data['overall_status'].isin(success_stati + failure_stati)].copy()
    data['target'] = data['overall_status'].apply(lambda x: 1 if x in success_stati else 0)
    data = data.drop(columns=['overall_status'])

    # --- Step 3: Handle Missing Values ---
    
    # 3a: Drop rows for 'sex' and 'enrollment_count'
    # The number of rows to drop are both under 3%, based on our EDA
    initial_rows = len(data)
    data.dropna(subset=['sex', 'enrollment_count'], inplace=True)
    print(f"Dropped {initial_rows - len(data)} rows with missing 'sex' or 'enrollment_count'.")

    # Define feature types
    categorical_features = ['phases', 'study_type', 'lead_sponsor_class', 'sex']
    numerical_features = ['enrollment_count', 'minimum_age', 'maximum_age']
    
    # 3b: Impute CATEGORICAL ('phases' only)
    # This will fill true NaN values, but leave the string "NA" untouched.
    # We do this because the "NA" string itself is usually used in trials or studies where phases are not relevant.
    # So we want to be able to distinguish between studies where the phases value was missing vs where it was not relevant. 
    data['phases'] = data['phases'].fillna('Unknown')

    # 3c: Impute NUMERICAL ('minimum_age' and 'maximum_age')
    transformation_rules = {'medians': {}}
    for col in ['minimum_age', 'maximum_age']:
        median_val = data[col].median()
        transformation_rules['medians'][col] = median_val
        data[col] = data[col].fillna(median_val)
        
    # Special fix for 'phases' if it's a list/ndarray
    data['phases'] = data['phases'].apply(lambda d: d[0] if isinstance(d, (list, np.ndarray)) and len(d) > 0 else d if not isinstance(d, (list, np.ndarray)) else 'Unknown')
    
    # --- Step 4: One-Hot Encode Categorical Features ---
    # handle_unknown='ignore' will create a column of all zeros for categories in the test set
    # that were not seen in the training set. This is a robust approach.
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(data[categorical_features])
    encoded_cols = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoder.transform(data[categorical_features]), columns=encoded_cols, index=data.index)
    transformation_rules['encoder'] = encoder

    # --- Step 5: Scale Numerical Features ---
    scaler = StandardScaler()
    scaler.fit(data[numerical_features])
    scaled_df = pd.DataFrame(scaler.transform(data[numerical_features]), columns=numerical_features, index=data.index)
    transformation_rules['scaler'] = scaler
    
    # Combine into the final processed DataFrame
    processed_train_df = pd.concat([scaled_df, encoded_df, data['target']], axis=1)

    print("Training data preprocessed successfully according to the specific plan.")
    
    return processed_train_df, transformation_rules