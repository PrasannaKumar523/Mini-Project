import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from flask import Flask, request, render_template
import pickle

# Load the dataset
data = pd.read_csv('cleaned_mumbai_dataset_new.csv').sample(10000, random_state=42)

# Define the feature set and target
X = data[['bhk', 'type', 'locality', 'area', 'region', 'status', 'age']]
y = data['price']

# Encode categorical features
categorical_features = ['type', 'locality', 'region', 'status', 'age']
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)


# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)

# Print R² scores to the console
print(f"Random Forest R² score: {r2_rf:.4f}")
print(f"Linear Regression R² score: {r2_lr:.4f}")

# Save both models
pickle.dump(rf_model, open('rf_model.pkl', 'wb'))
pickle.dump(lr_model, open('lr_model.pkl', 'wb'))

# Create Flask app
app = Flask(__name__)

# Load the models
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
lr_model = pickle.load(open('lr_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the user
    input_data = {
        'bhk': int(request.form['bhk']),
        'type': request.form['type'],
        'locality': request.form['locality'],
        'area': float(request.form['area']),
        'region': request.form['region'],
        'status': request.form['status'],
        'age': request.form['age']
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    input_df = pd.get_dummies(input_df, columns=['type', 'locality', 'region', 'status', 'age'], drop_first=True)

    # Handle missing columns
    missing_cols = list(set(X.columns) - set(input_df.columns))
    missing_df = pd.DataFrame(0, index=input_df.index, columns=missing_cols)
    input_df = pd.concat([input_df, missing_df], axis=1)

    # Reorder columns to match the training data
    input_df = input_df[X.columns]

    # Predict using both models
    rf_prediction = rf_model.predict(input_df)[0]
    lr_prediction = lr_model.predict(input_df)[0]

    # Compare R² scores and choose the model with the higher score
    if r2_rf >= r2_lr:
        prediction = rf_prediction
    else:
        prediction = lr_prediction

    # Return the prediction to the user
    return render_template('index.html', prediction_text=f'Predicted Price: {prediction:.2f} Cr')

if __name__ == '__main__':
    app.run(debug=True)