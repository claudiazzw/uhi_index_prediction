# Train the Random Forest model on the training data
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train,y_train)