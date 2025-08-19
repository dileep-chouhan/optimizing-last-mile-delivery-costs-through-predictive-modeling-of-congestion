import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# --- 1. Synthetic Data Generation ---
# Generate synthetic data for delivery routes, congestion levels, and delivery times.
np.random.seed(42)  # for reproducibility
num_deliveries = 100
data = {
    'Route': np.random.choice(['A', 'B', 'C'], size=num_deliveries),
    'Distance': np.random.uniform(5, 20, size=num_deliveries), # in km
    'Congestion_Level': np.random.randint(1, 10, size=num_deliveries), # 1-low, 10-high
    'Delivery_Time': np.random.uniform(30, 120, size=num_deliveries) # in minutes
}
df = pd.DataFrame(data)
# Add some noise to the data to make it more realistic.
df['Delivery_Time'] += np.random.normal(0, 10, size=num_deliveries)
df['Delivery_Time'] = df['Delivery_Time'].clip(min=0) # Ensure time is non-negative
# --- 2. Data Cleaning and Feature Engineering ---
# No significant cleaning needed for synthetic data, but we can create a combined feature.
df['Congestion_Distance'] = df['Congestion_Level'] * df['Distance']
# --- 3. Predictive Modeling ---
# Build a linear regression model to predict delivery time based on congestion and distance.
X = df[['Congestion_Distance', 'Distance']]
y = df['Delivery_Time']
model = LinearRegression()
model.fit(X, y)
# --- 4. Model Evaluation (Simple example) ---
r_sq = model.score(X, y)
print(f"coefficient of determination: {r_sq}")
# --- 5. Visualization ---
# Plot the relationship between congestion_distance and delivery time
plt.figure(figsize=(10, 6))
plt.scatter(df['Congestion_Distance'], df['Delivery_Time'])
plt.title('Delivery Time vs. Congestion Distance')
plt.xlabel('Congestion Distance')
plt.ylabel('Delivery Time (minutes)')
plt.grid(True)
plt.tight_layout()
output_filename = 'congestion_vs_delivery_time.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Plot the model's prediction against actual values.
plt.figure(figsize=(10,6))
plt.scatter(y, model.predict(X))
plt.xlabel("Actual Delivery Time")
plt.ylabel("Predicted Delivery Time")
plt.title("Actual vs Predicted Delivery Time")
plt.plot([min(y),max(y)], [min(y),max(y)], color='red') #Line of perfect prediction
plt.grid(True)
plt.tight_layout()
output_filename2 = 'actual_vs_predicted.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")
# --- 6.  Further Analysis (Illustrative) ---
#Example of how to use the model for prediction (on new data)
new_data = pd.DataFrame({'Distance':[10,15,8], 'Congestion_Level':[5,2,9]})
new_data['Congestion_Distance'] = new_data['Congestion_Level']*new_data['Distance']
predicted_times = model.predict(new_data[['Congestion_Distance', 'Distance']])
print("\nPredicted delivery times for new data:")
print(predicted_times)