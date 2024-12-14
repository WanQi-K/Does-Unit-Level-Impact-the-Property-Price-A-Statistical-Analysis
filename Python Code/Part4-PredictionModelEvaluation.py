# Add predicted prices to the dataset
def add_predicted_prices(row):
    square_feet = row['Square Feet']
    level = row['Unit Level']
    predicted_price, _, _ = predict_price_ci_with_level(square_feet, level, model_with_level, SE_with_level, t_critical_with_level)
    return predicted_price

# Apply the function row by row
InderaSubang_data['Predicted Price'] = InderaSubang_data.apply(add_predicted_prices, axis=1)

import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot of actual vs. predicted prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=InderaSubang_data['Transaction Price'], y=InderaSubang_data['Predicted Price'])

# Add a line to indicate perfect prediction
max_price = max(InderaSubang_data['Transaction Price'].max(), InderaSubang_data['Predicted Price'].max())
plt.plot([0, max_price], [0, max_price], color='red', linestyle='--', label='Perfect Prediction')

# Customize the plot
plt.title('Actual Transacted Price vs. Predicted Price', fontsize=14)
plt.xlabel('Actual Transacted Price (RM)', fontsize=12)
plt.ylabel('Predicted Price (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Calculate residuals
InderaSubang_data['Residual'] = InderaSubang_data['Transaction Price'] - InderaSubang_data['Predicted Price']

# Residual plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=InderaSubang_data['Transaction Price'], y=InderaSubang_data['Residual'])
plt.axhline(0, color='red', linestyle='--', label='Zero Residual')

# Customize the plot
plt.title('Residuals vs. Actual Transacted Price', fontsize=14)
plt.xlabel('Actual Transacted Price (RM)', fontsize=12)
plt.ylabel('Residual (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
