# Wildfire_Smoke_ML_Prediction_2024
**Author:** Michael R. Martin

**Initially Uploaded on GitHub: 11/10/2024**

## Purpose:  
The purpose of this program is to predict *Air Pollutant Levels* (**PM2.5 levels**) in the atmosphere, specifically targeting wildfire smoke emission impacts using various machine learning techniques, including a newly added LSTM model for time series prediction. It achieves this by analyzing a range of environmental and meteorological features that influence wildfire smoke behavior and **PM2.5** emissions. The features include vapor pressure deficit, maximum temperature, precipitation, and other drought and moisture indices.

Key aspects of the program include:

1. **Data Handling and Visualization**: Historical big data from 1996 to 2020, which is loaded and visualized for trend observation and correlations in features affecting wildfire smoke levels.
2. **Data Processing**: Input variables undergo rescaling and reshaping through statistical analysis to optimize for machine learning models in dictionary.
3. **Machine Learning and Deep Learning Model Training**: The program employs multiple Machine Learning techniques through Gradient Boosting, AdaBoost, Random Forest, Decision Tree, K-Nearest Neighbors, Ridge, and SVR models, with an additional LSTM model for time series data. The updated dictionary is as follows:
   ```python
   models = {
       'Gradient Boost': GradientBoostingRegressor(random_state=0),
       'Ada Boost': AdaBoostRegressor(random_state=0),
       'Random Forest': RandomForestRegressor(random_state=0),
       'Decision Tree': DecisionTreeRegressor(random_state=0),
       'KNN': KNeighborsRegressor(),
       'Ridge': Ridge(random_state=0),
       'SVR': SVR(),
       'LSTM': 'lstm'  # Placeholder for LSTM
   }
   ```
   **LSTM Model**:  
   For LSTM, data is prepared by converting training inputs into tensors, adding a time dimension. A batch size of 256 is used, with **Mean Squared Error** (MSE) as the loss function and **Adam** as the optimizer, with the MSE recorded and plotted per epoch to visualize performance.
4. **Model Performance Evaluation**: Each model is trained on the training dataset and tested on test data, with performance metrics generated to assess predictive accuracy and model suitability. SHAPley values are also applied to interpret feature importance for each model.

This program encompasses starting code created for wildfire smoke prediction using several Machine Learning methods, and includes the ability to save and load trained models, allowing for easy reuse of specific versions, including the LSTM model.

**Input variables:**  
- vapor pressure deficit (VPD)  
- Temperature (max)  
- Precipitation  
- Potential evapotranspiration (PET)  
- evapotranspiration (ET)  
- Palmer Drought Severity Index (PDSI)  
- Evaporative Demand Drought Index (EDDI)  
- soil moisture  
- water equivalent drought index (SWEI)  
- **Output variable**: Wildfire Smoke Emissions (**PM2.5**)

## Code Info:  
After input variables are initialized, input variables and emission big data are read (from 1996 to 2020) before being visualized. After rescaling and reshaping through statistical analysis, several ML models are trained with the training datasets. The updated dictionary of models is:
```python
'Gradient Boost': GradientBoostingRegressor(random_state=0),
'Ada Boost': AdaBoostRegressor(random_state=0),
'Random Forest': RandomForestRegressor(random_state=0),
'Decision Tree': DecisionTreeRegressor(random_state=0),
'KNN': KNeighborsRegressor(),
'Ridge': Ridge(random_state=0),
'SVR': SVR(),
'LSTM': 'lstm'  # Placeholder for LSTM
```

For the LSTM model, additional coding is included to prepare data, run a training loop, track MSE, and generate a plot for epoch-based MSE. Model statistics, such as linear regression statistics, are calculated and displayed after training and predictions.
