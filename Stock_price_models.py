import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt
import seaborn
import os
import joblib
from sklearn.impute import SimpleImputer
import locale

dataset_dir = 'Datasets/'
files = os.listdir(dataset_dir)

data = pd.DataFrame()  # Empty DataFrame to hold the data from all files
models = {}

save_dir = 'models/'

for file in files:
    file_path = os.path.join(dataset_dir, file)
    if file.endswith('.csv'):
        df = pd.read_csv(file_path)
        #data = data.append(df, ignore_index=True)
        df = df.drop(['Date'], axis=1)

        for column in df.columns:
            if df[column].dtype == object and df[column].str.contains(',').any():
                # Remove the thousand separators from the column
                df[column] = df[column].str.replace(',', '')
                df[column] = pd.to_numeric(df[column], errors='coerce')

        column_names = df.columns
        
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant')
        X_imputed = imputer.fit_transform(df)
        df = X_imputed[~np.isnan(X_imputed).any(axis=1)]
        df = pd.DataFrame(df, columns=column_names)
        

        X = df.drop(['Close', 'Adj Close'], axis=1)
        y = df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model_name = os.path.splitext(file)[0]
        model = LinearRegression()
        model.fit(X_train, y_train)
        models[model_name] = {'model': model, 'X_test': X_test, 'y_test': y_test}
        #print(X_test)
        y_pred = model.predict(X_test)
        comparision = pd.DataFrame({'Predicted Value': y_pred, 'Actual Value': y_test})
        print(comparision)

for model_name, data in models.items():
    model = data['model']
    
    # Save the model with the name of the file
    model_file_path = os.path.join(save_dir, model_name + '.joblib')
    joblib.dump(model, model_file_path)
    
    print(f"Model '{model_name}' saved as '{model_file_path}'\n")



# Step 11: Deploy and use the models (optional)
# Assuming you want to load the saved Model 1 and make predictions
# model_file_name = 'TSLA_model.joblib'  # Replace 'file_name' with the desired model name
# loaded_model = joblib.load(model_file_name)
# print(loaded_model.predict([[20.61, 21, 20.5, 243400]]))

# Make predictions on new data
# new_data = pd.read_csv('path_to_new_data.csv')
# predictions = loaded_model.predict(new_data)

