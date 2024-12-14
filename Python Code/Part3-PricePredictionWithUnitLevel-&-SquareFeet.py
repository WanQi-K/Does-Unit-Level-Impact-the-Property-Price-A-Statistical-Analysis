def predict_price_ci_with_level(square_feet, level, model, SE, t_critical):
    example_input = pd.DataFrame({'Square Feet': [square_feet], 'Unit Level': [level]})
    predicted_price = model.predict(example_input)[0]
    
    # Confidence Interval
    margin_of_error = t_critical * SE
    lower_bound = predicted_price - margin_of_error
    upper_bound = predicted_price + margin_of_error
    
    return predicted_price, lower_bound, upper_bound

# Add 'Level' to confidence interval calculations
confidence_level = 0.90  # 90% CI
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
