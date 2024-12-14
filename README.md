# Does-Unit-Level-Impact-the-Property-Price-A-Statistical-Analysis

# Intro : High-Rise Market Trends in Malaysia

The COVID-19 pandemic reshaped the property market in Malaysia, particularly in how people value their living spaces. Since the Movement Control Order (MCO) in 2020, high-rise properties have gained favor due to their stricter access controls, which enhance health and safety.

Besides the usual considerations of budget and location, a key decision now is whether to choose a high-rise or landed property. Within the high-rise segment, higher-floor units are typically priced higher, as they are associated with better views, privacy, and reduced noise from street level.

However, before assuming that higher floors are always worth the premium, several factors must be considered:

- **Views:** While higher floors are often marketed as offering better views, these views may change over time due to new developments or infrastructure projects.
- **Noise pollution:** Contrary to popular belief, noise can travel upwards, which means noise levels may not always decrease with height.

### Related Articles

If you’re interested in exploring more property-related analyses, check out my previous articles:

1. [**Understanding Property Market Trends**](https://medium.com/@kwanqi.yt/real-estate-analysis-with-python-cfe7eb4cbd88) – A data-driven property analysis for properties in Malaysia
2. **Is Your Property Overpriced?** – A look into identifying overpriced or undervalued properties using data

### Dataset Overview

For this analysis, we used data from the **National Property Information Centre (NAPIC)**, which provides transactional data for Malaysian properties.

### Key Details:

- **Source:** [NAPIC Open Data](https://napic2.jpph.gov.my/en/open-sales-data?category=36&id=241)
- **Time Period:** Transactions from 2021 to mid-2024.
- **Data Points:** Property area, name, transaction date, size (sq. ft.), and transaction price.
- **Property Coverage :** Indera Subang Condominium

## Does the Unit Level Really Impact the Property Price?

To address this question, I :

1. Evaluated the relevance of the `Unit Level` feature using **R-squared** values.
2. Built a prediction model incorporating `Unit Level` and `Square Feet` to assess its impact on price forecasting accuracy.

## Feature Relevance Evaluation

We compared two models:

- Model 1: Uses `Square Feet` as the sole predictor.
- Model 2: Includes both `Square Feet` and `Unit Level`.

```python
# Add 'Level' feature to the data
# Split data with and without 'Level' feature
X_with_level = InderaSubang_data[["Square Feet", "Unit Level"]]
X_without_level = InderaSubang_data[["Square Feet"]]
y = InderaSubang_data["Transaction Price"]

# Train-Test Split for both scenarios
X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(X_with_level, y, test_size=0.2, random_state=42)
X_train_without, X_test_without, y_train_without, y_test_without = train_test_split(X_without_level, y, test_size=0.2, random_state=42)

# Train models
model_with_level = LinearRegression()
model_with_level.fit(X_train_with, y_train_with)

model_without_level = LinearRegression()
model_without_level.fit(X_train_without, y_train_without)

# Calculate R-squared
r2_with_level = model_with_level.score(X_test_with, y_test_with)
r2_without_level = model_without_level.score(X_test_without, y_test_without)

# Display R-squared values
print(f"R-squared without 'Unit Level': {r2_without_level:.4f}")
print(f"R-squared with 'Unit Level': {r2_with_level:.4f}")
```

Result : 

![17 1](https://github.com/user-attachments/assets/897b7c1c-0877-45fb-9200-744d3c965f9e)


### What is R-squared?

The **R-squared value** measures how well the independent variables explain the variability of the dependent variable.

- A higher R-squared value indicates a stronger correlation between the features (e.g., `Unit Level`) and the outcome (e.g., transaction price).
- In this analysis, including `Unit Level` improved the R-squared from **0.332** to **0.475**, demonstrating a higher impact of `Unit Level` on property prices in this dataset.

## Prediction Analysis: Estimating Property Prices

Let’s consider a scenario: You’re interested in purchasing a unit in **Indera Subang Condominium** with the following details:

- **Size:** 1711 sq. ft.
- **Level:** 8

What price range should you expect, based on the model?

```python
def predict_price_ci_with_level(square_feet, level, model, SE, t_critical):
    example_input = pd.DataFrame({'Square Feet': [square_feet], 'Unit Level': [level]})
    predicted_price = model.predict(example_input)[0]
    
    # Confidence Interval
    margin_of_error = t_critical * SE
    lower_bound = predicted_price - margin_of_error
    upper_bound = predicted_price + margin_of_error
    
    return predicted_price, lower_bound, upper_bound

# Add 'Level' to confidence interval calculations
residuals_with_level = y_train_with - model_with_level.predict(X_train_with)
std_residuals_with_level = np.std(residuals_with_level)
N_with_level = len(y_train_with)
SE_with_level = std_residuals_with_level / np.sqrt(N_with_level)
t_critical_with_level = t.ppf((1 + confidence_level) / 2, N_with_level - 2)

# Example prediction with 'Level'
square_feet = 1711 #Input Square Feet
level = 8  # Input level
predicted_price_with_level, lower_bound_with_level, upper_bound_with_level = predict_price_ci_with_level(
    square_feet, level, model_with_level, SE_with_level, t_critical_with_level
)

# Display results
print(f"Predicted Price for {square_feet} square feet on Level {level}: RM {predicted_price_with_level:,.0f}")
print(f"90% Confidence Interval: RM {lower_bound_with_level:,.0f} - RM {upper_bound_with_level:,.0f}")

```

Result : 

![17 2](https://github.com/user-attachments/assets/0933608b-ed3d-4199-90c3-d39d1c7d1e14)


**Interpretation:** The model predicts a transaction price range between **RM 672,433** and **RM 718,833** with 90% confidence.

# Findings & Insights

### Comparison with Actual Transaction Data

```python
InderaSubang_data_Filtered = df[(df['Square Feet'] == 1711) & (df['Unit Level'] == 8)]
```

![17 3](https://github.com/user-attachments/assets/7f02bf7a-2d96-44d0-b8ff-e3ca200811a9)


In this specific case, the actual transacted price for a similar unit was **RM 760,000**. Since this price exceeds the predicted range, it suggests that the property may be **overvalued** based on historical trends.

### Price Range Variance

The analysis also revealed that:

- A model using only `Square Feet` had a variance of RM **64,081 ([here](https://medium.com/p/474acd9a68f4/edit))**
- Adding `Unit Level` reduced the variance to RM **46,400**, indicating improved accuracy.

## Diagnosis on the Prediction Model

```python
# Add predicted prices to the dataset
def add_predicted_prices(row):
    square_feet = row['Square Feet']
    level = row['Unit Level']
    predicted_price, _, _ = predict_price_ci_with_level(square_feet, level, model_with_level, SE_with_level, t_critical_with_level)
    return predicted_price

# Apply the function row by row
InderaSubang_data['Predicted Price'] = InderaSubang_data.apply(add_predicted_prices, axis=1)
```

## Scatterplot

The scatterplot shows the actual transaction prices versus the model's predicted prices for the Indera Subang dataset.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot of actual vs. predicted prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=InderaSubang_data['Transaction Price'], y=InderaSubang_data['Predicted Price'])

# Add a line to indicate perfect prediction
max_price = max(InderaSubang_data['Transaction Price'].max(), InderaSubang_data['Predicted Price'].max())
plt.plot([0, max_price], [0, max_price], color='red', linestyle='--', label='Perfect Prediction')

# Customize the plot
plt.title('Transaction Price vs. Predicted Price', fontsize=14)
plt.xlabel('Transaction Price (RM)', fontsize=12)
plt.ylabel('Predicted Price (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
```
![17 4](https://github.com/user-attachments/assets/987ff8f6-39f0-4670-b681-1bec22f4c12d)


### Key Observations :

- **General Trend:** Most points align closely with the red dashed line, indicating that the model performs reasonably well in predicting transaction prices.
- **Deviations:** Some data points deviate from the line, particularly in the higher price range, suggesting the model slightly underpredicts for premium properties.
- **Clusters:** There are visible clusters where the model predicts similarly for properties in the mid-range price category, reflecting a potential consistency in its accuracy for this group.

**Summary :** The scatterplot suggests that while the model performs well overall, it may require adjustments to improve accuracy for higher-priced properties. More data is needed to train the model to improve the accuracy. 

## Residual Plot

The residual plot displays the difference between actual and predicted prices (residuals) versus the actual transaction prices.

```python
# Calculate residuals
InderaSubang_data['Residual'] = InderaSubang_data['Transaction Price'] - InderaSubang_data['Predicted Price']

# Residual plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=InderaSubang_data['Transaction Price'], y=InderaSubang_data['Residual'])
plt.axhline(0, color='red', linestyle='--', label='Zero Residual')

# Customize the plot
plt.title('Residuals vs. Transaction Price', fontsize=14)
plt.xlabel('Transaction Price (RM)', fontsize=12)
plt.ylabel('Residual (RM)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
```

![17 5](https://github.com/user-attachments/assets/fa74c29f-8379-45e1-8dac-4ff983a2fecd)


### Key Observation :

- **Distribution:** Residuals are mostly centered around 0, indicating an unbiased model with errors evenly distributed across price ranges.
- **Patterns:** A slight pattern emerges for higher transaction prices, where residuals are more dispersed, suggesting increasing error variance as prices rise (heteroscedasticity).
- **Outliers:** A few points with large residuals indicate the model struggled to predict these cases accurately, potentially due to unique property features not captured in the dataset.

**Summary :** The residual plot supports the assumption of linearity but highlights areas for improvement, such as investigating outliers for better predictive performance.

---

## Conclusion

Our analysis shows that `Unit Level` significantly impacts property prices in high-rise developments like Indera Subang Condominium. By incorporating this feature, we improved the model's predictive accuracy and narrowed price ranges, providing better insights for buyers.

This prediction model has significant room for improvement, particularly due to the limited dataset available for this property. While it provides a helpful guideline, it should be used as a starting point for further research and analysis before making a final decision on purchasing a home.

If you're considering a purchase, such an analysis can help you determine whether the asking price aligns with market trends—helping you make informed decisions and negotiate effectively.

Next article, we will be exploring the actual listing with asking price and compare against our model, to see if there’s any property is undervalued or overpriced.
