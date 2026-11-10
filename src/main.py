import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# To ensure outputs file exists before graph image
os.makedirs("outputs", exist_ok=True)

# 1. Getting the dataset from the data folder
df = pd.read_csv("data/airfoil_self_noise.csv")
df.columns = ['Frequency', 'Angle_of_attack', 'Chord_length', 'Velocity', 'Displacement_thickness', 'SSPL']

print("Dataset shape:", df.shape)
print(df.head())

# 2. Defining features (X) and target (y)
X = df[['Frequency', 'Angle_of_attack', 'Chord_length', 'Velocity', 'Displacement_thickness']]
y = df['SSPL']

# 3 Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

#4. Fitting the model
lr = LinearRegression()
lr.fit(X_train, y_train)
print("\nCoefficients:", np.round(lr.coef_, 3))
print("Intercept:", round(lr.intercept_, 3))

# 5. Predict on test set
y_pred = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", round(mae, 3))
print("R2 Score:", round(r2, 3))
print("RMSE:", round(rmse, 3))

# 6.Ploting
plt.figure()
plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", label="prediction")  # Ideal 45-degree line
plt.xlabel("Actual SSPL (dB)")
plt.ylabel("Predicted SSPL (dB)")
plt.title("Actual vs Predicted Sound Pressure Level")
plt.legend()

# Save the graph
plt.savefig("outputs/predicted_vs_actual.png")
print("\nPlot saved to outputs/predicted_vs_actual.png")
