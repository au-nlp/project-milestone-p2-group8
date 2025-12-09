
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
    
    # We clip ages to reasonable human limits (e.g., 0 to 120 years)
    # This prevents outliers from distorting the StandardScaler for the Fusion Model.
    data['minimum_age'] = data['minimum_age'].clip(0, 120)
    data['maximum_age'] = data['maximum_age'].clip(0, 120)

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


# preprocess function that preprocesses our testing data data and applies the transformation rules

def transform_test_data(test_df: pd.DataFrame, rules: dict):
    """
    Applies the learned preprocessing rules to the raw test DataFrame.
    
    Args:
        test_df: The raw testing pandas DataFrame.
        rules: The dictionary of transformation rules learned from the training data.
        
    Returns:
        A cleaned and preprocessed test DataFrame ready for evaluation.
    """
    data = test_df.copy()

    # --- Step 1 & 2: Feature Selection and Target Creation ---
    # (Same initial steps as the training function)
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

    # --- Step 3: Handle Missing Values using LEARNED Rules ---
    # Drop rows first, mirroring the training process
    data.dropna(subset=['sex', 'enrollment_count'], inplace=True)
    
    categorical_features = ['phases', 'study_type', 'lead_sponsor_class', 'sex']
    numerical_features = ['enrollment_count', 'minimum_age', 'maximum_age']
    
    # Impute CATEGORICAL ('phases' only)
    data['phases'] = data['phases'].fillna('Unknown')

    # NEW: Cap Age Outliers
    data['minimum_age'] = data['minimum_age'].clip(0, 120)
    data['maximum_age'] = data['maximum_age'].clip(0, 120)

    # Impute NUMERICAL using the SAVED medians from the rules dictionary
    for col in ['minimum_age', 'maximum_age']:
        data[col] = data[col].fillna(rules['medians'][col])
        
    # Special fix for 'phases' if it's a list/ndarray
    data['phases'] = data['phases'].apply(lambda d: d[0] if isinstance(d, (list, np.ndarray)) and len(d) > 0 else d if not isinstance(d, (list, np.ndarray)) else 'Unknown')

    # --- Step 4: One-Hot Encode Categorical Features using LEARNED Encoder ---
    # Retrieve the fitted encoder from the rules
    encoder = rules['encoder']
    encoded_cols = encoder.get_feature_names_out(categorical_features)
    # APPLY the learned transformation
    encoded_df = pd.DataFrame(encoder.transform(data[categorical_features]), columns=encoded_cols, index=data.index)

    # --- Step 5: Scale Numerical Features using LEARNED Scaler ---
    # Retrieve the fitted scaler from the rules
    scaler = rules['scaler']
    # APPLY the learned transformation
    scaled_df = pd.DataFrame(scaler.transform(data[numerical_features]), columns=numerical_features, index=data.index)
    
    # Combine into the final processed DataFrame
    processed_test_df = pd.concat([scaled_df, encoded_df, data['target']], axis=1)

    print("Test data transformed successfully using training set rules.")
    
    return processed_test_df

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import resample

# dataset class for pytorch - we are taking our parqet and making sure it gets processed so that pytorch can read it

import torch
from torch.utils.data import Dataset

class ClinicalTrialDataset(Dataset):
    def __init__(self, structured_features, embeddings, labels=None):
        """
        Args:
            structured_features (np.array): The standardized metadata (23 columns).
            embeddings (np.array): The BERT embeddings (1536 columns).
            labels (np.array, optional): The target (0 or 1). None for test set.
        """
        # We convert numpy arrays to PyTorch Tensors (Float32 is standard for ML)
        self.structured = torch.tensor(structured_features, dtype=torch.float32)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        
        # Labels are optional (we might not have them for a pure inference set, though usually we do)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) # Shape becomes (N, 1)
        else:
            self.labels = None

    def __len__(self):
        # Question 1: How long is the dataset?
        return len(self.structured)

    def __getitem__(self, idx):
        # Question 2: Get me the data for row 'idx'
        struct_sample = self.structured[idx]
        embed_sample = self.embeddings[idx]
        
        if self.labels is not None:
            label_sample = self.labels[idx]
            return struct_sample, embed_sample, label_sample
        else:
            return struct_sample, embed_sample
        

# this is our model training loop - maybe i will move this to the main notebook..

import torch.optim as optim

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
    """
    Args:
        model: The MultimodalNet instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs: How many times to loop through the full dataset.
        learning_rate: How big the steps are during optimization.
    """
    
    # 1. Setup Device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device) # Move model to the GPU
    
    # 2. Define Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss() # The standard for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track metrics
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train() # Set model to training mode (enables Dropout)
        running_loss = 0.0
        
        # --- Training Phase ---
        for batch_struct, batch_emb, batch_labels in train_loader:
            # Move data to GPU
            batch_struct = batch_struct.to(device)
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)
            
            # Zero the gradients (PyTorch accumulates them by default)
            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(batch_struct, batch_emb)
            
            # Calculate Loss
            loss = criterion(outputs, batch_labels)
            
            # Backward Pass & Optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode (disables Dropout)
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): # Don't calculate gradients during validation (saves memory)
            for batch_struct, batch_emb, batch_labels in val_loader:
                batch_struct = batch_struct.to(device)
                batch_emb = batch_emb.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_struct, batch_emb)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                # Calculate Accuracy (for monitoring)
                # Since we use BCEWithLogits, we use Sigmoid to get probability > 0.5
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # Store metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
              
    return history


# dataset balancing code
from sklearn.utils import resample

def balance_raw_data(df):
    """
    Takes the RAW training dataframe.
    Returns a new dataframe where Success (1) and Failure (0) counts are equal.
    """
    # 1. Identify the classes using the specific strings from P2
    success_stati = ["COMPLETED"]
    failure_stati = ["TERMINATED", "WITHDRAWN", "SUSPENDED"]
    
    # Filter for only definitive outcomes (just in case)
    df_clean = df[df['overall_status'].isin(success_stati + failure_stati)].copy()
    
    # Create a temporary target just for splitting
    df_clean['temp_target'] = df_clean['overall_status'].apply(lambda x: 1 if x in success_stati else 0)
    
    # Separate
    df_majority = df_clean[df_clean.temp_target == 1]
    df_minority = df_clean[df_clean.temp_target == 0]
    
    # 2. Downsample Majority
    print(f"Original counts - Success: {len(df_majority)}, Failure: {len(df_minority)}")
    
    df_majority_downsampled = resample(
        df_majority, 
        replace=False,    # No duplicates
        n_samples=len(df_minority), # Match the minority count
        random_state=42
    )
    
    # 3. Combine
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    
    # Shuffle so we don't have all successes then all failures
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced counts - Success: {len(df_majority_downsampled)}, Failure: {len(df_minority)}")
    return df_balanced