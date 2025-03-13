from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import kagglehub

app = Flask(__name__)

# Download and load the dataset
path = kagglehub.dataset_download("zhaoyingzhu/heartcsv")
heart_data_path = path + "/heart.csv"
heart_data = pd.read_csv(heart_data_path)  # Corrected line

heart_data.columns = heart_data.columns.str.strip()

if 'Unnamed: 0' in heart_data.columns:
    heart_data = heart_data.drop(['Unnamed: 0'], axis=1)

categorical_columns = ['ChestPain', 'Thal', 'AHD']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(heart_data[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

heart_data = heart_data.drop(categorical_columns, axis=1)
heart_data = pd.concat([heart_data, encoded_df], axis=1)

imputer = SimpleImputer(strategy='mean')
heart_data_imputed = pd.DataFrame(imputer.fit_transform(heart_data), columns=heart_data.columns)

X = heart_data_imputed.drop(['AHD_Yes'], axis=1)
y = heart_data_imputed['AHD_Yes']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

def predict_heart_condition(input_data):
    categorical_data = [input_data.pop('ChestPain'), input_data.pop('Thal'), 'Yes']
    categorical_encoded = encoder.transform([categorical_data])[0][:3]
    input_data = list(input_data.values())
    input_data.extend(categorical_encoded)

    full_input_data = np.zeros(scaler.n_features_in_)
    full_input_data[:len(input_data)] = input_data
    input_data_scaled = scaler.transform([full_input_data])

    prediction = model.predict(input_data_scaled)[0]
    return "Defected Heart" if prediction == 1 else "Healthy Heart"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'Age': int(request.form['age']),
            'Sex': int(request.form['sex']),
            'RestBP': int(request.form['restbp']),
            'Chol': int(request.form['chol']),
            'Fbs': int(request.form['fbs']),
            'RestECG': int(request.form['restecg']),
            'MaxHR': int(request.form['maxhr']),
            'ExAng': int(request.form['exang']),
            'Oldpeak': float(request.form['oldpeak']),
            'Slope': int(request.form['slope']),
            'Ca': int(request.form['ca']),
            'ChestPain': request.form['chestpain'],
            'Thal': request.form['thal']
        }
        result = predict_heart_condition(input_data)
        return render_template('index.html', result=result)
    return render_template('index.html', result='')

if __name__ == '__main__':
    app.run(debug=True)
