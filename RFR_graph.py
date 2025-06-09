import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to load dataset (replace with your actual file)
def load_data():
    data = pd.read_csv('cleaned_mumbai_dataset_new.csv').sample(500, random_state=42)
    return data

# Random Forest Regression and plot with cleaner graph
def simple_random_forest_plot():
    data = load_data()

    # Features and target
    X = data[['bhk', 'type', 'locality', 'area', 'region', 'status', 'age']]  # Categorical and numerical
    y = data['price']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing: OneHotEncode the categorical features ('type', 'locality', 'region', 'status')
    categorical_features = ['type', 'locality', 'region', 'status','age']
    numerical_features = ['bhk', 'area']

    # Define the ColumnTransformer with handle_unknown='ignore'
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ]
    )

    # Create pipeline with preprocessing and random forest regressor
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred = pipeline.predict(X_test)

    # Plotting actual vs predicted prices with cleaner visualization
    plt.figure(figsize=(10, 6))

    # Plot points with transparency to avoid clutter
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, s=40, label='Predicted vs Actual')

    # Plot a perfect-fit line (diagonal line) for reference
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Perfect Fit')

    # Labeling axes and title
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Random Forest Regression - Actual vs Predicted Prices')

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    simple_random_forest_plot()
