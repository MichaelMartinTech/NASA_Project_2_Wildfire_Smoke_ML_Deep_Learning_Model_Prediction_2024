# NASA-Funded Project: Machine Learning and Deep Learning-Based Wildfire Prediction using Big Data
**Author:** Michael R. Martin

**Initially Uploaded on GitHub: 11/10/2024**

## Purpose:  
I have developed a research proposal to enhance the predictability of complex environmental issues. This project is funded by NASA, where I designed, developed, and implemented this project to predict *smoke emissions from wildfires* (**PM2.5 levels**) based on various machine learning techniques, including *deep learning* / *Recurrent Neural Networks* (RNN). This project used ensemble big data to analyze a range of environmental and meteorological features as input of predictive models including high-resolution input data of vapor pressure deficit, maximum temperature, precipitation, and other drought and moisture indices to predict smoke emissions as the output of the models. The results will be presented in the upcoming NASA statewide meeting and will be published in a peer-reviewed publication.

## Key aspects of the program include:

1. **Data Handling and Visualization**: Historical big data from 1996 to 2020, which is loaded and visualized for trend observation and correlations in features affecting wildfire smoke levels.
2. **Data Processing**: Input variables undergo rescaling and reshaping through statistical analysis to optimize for machine learning models in dictionary.
3. **Machine Learning and Deep Learning Model Training**: The program employs multiple Machine Learning techniques through Gradient Boosting, AdaBoost, Random Forest, Decision Tree, K-Nearest Neighbors, Ridge, and SVR models, with an additional LSTM model for time series data.
   
   **LSTM Model**:  
   For LSTM, data is prepared by converting training inputs into tensors, adding a time dimension. A batch size of 256 is used, with **Mean Squared Error** (MSE) as the loss function and **Adam** as the optimizer, with the MSE recorded and plotted per epoch to visualize performance.
5. **Model Performance Evaluation**: Each model is trained on the training dataset and tested on test data, with performance metrics generated to assess predictive accuracy and model suitability. SHAPley values are also applied to interpret feature importance for each model.

This program encompasses starting code created for wildfire smoke prediction using several Machine Learning methods, and includes the ability to save and load trained models, allowing for easy reuse of specific versions, including the LSTM model.

## Code Info:  
After input variables are initialized, input variables and emission big data are read (from 1996 to 2020) before being visualized. After rescaling and reshaping through statistical analysis, several ML models are trained with the training datasets. The updated dictionary is as follows:
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
After input variables are initialized, input variables and emission big data are read (from 1996 to 2020) before being visualized. After rescaling and reshaping through statistical analysis, several ML models are trained with the training datasets. Data is prepared, a training loop is run, MSE tracked, and a plot is generated for epoch-based MSE. Model statistics, such as linear regression statistics, are calculated and displayed after training and predictions.
