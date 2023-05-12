# ML_Hackrush-23

## Forecasting stocks price using AI

Our team, predict.me(), participated in the ML Hackrush-23 challenge to forecast the stock prices of three subsidiaries of TGD, namely TGD Consultancy, TGD Automobiles, and TGD Power. We leveraged AI and advanced algorithms to analyze the provided dataset and make accurate predictions for maximizing profits in the stock market. Our solution incorporated various techniques, including linear regression with hyperparameter tuning, ensemble learning, dimensionality reduction using PCA, and LSTM stacking.

## Approach
### From data extraction all the way to model selection

#### 1. Data Analysis: 
We started by performing exploratory data analysis (EDA) to understand the characteristics and interdependencies of the dataset. With the help of heatmaps, we identified high correlations between several factors, indicating the possibility of dimensionality reduction using PCA techniques.

#### 2. Dimensionality Reduction: 
We applied PCA (Principal Component Analysis) to reduce the dimensionality of the dataset and remove features that had high correlation and interdependencies. This step helped improve the prediction of each target column.

#### 3. Linear Regression:
We employed linear regression models with hyperparameter tuning to make initial predictions. During this process, we dropped multiple feature columns with high correlations to reduce redundancy and improve model performance.

#### 4. Ensemble Learning:
To further enhance the predictive power, we employed ensemble learning techniques. We created an ensemble of decision trees, random forest, SVM (Support Vector Machines), and Lasso regression models in the first layer of the ensemble. Then, we used an MLP (Multi-Layer Perceptron) layer to optimize the weight distribution to all the models in the first layer.

#### 5. LSTM Stacking
Based on extensive experimentation, LSTM stacking emerged as the most effective approach for multi-task, multi-variate time-series forecasting. LSTM models excel in capturing complex temporal patterns and dependencies in the data, making them suitable for forecasting stock prices. By stacking multiple LSTM layers, we were able to learn and leverage both short-term and long-term dependencies, resulting in improved accuracy and robustness.

## Results:
Our final submission achieved significant improvements in prediction accuracy, enabling us to secure a top position on the leaderboard. We applied appropriate evaluation metrics, such as RMSE (Root Mean Square Error), to assess the performance of our models. The LSTM stacking technique demonstrated superior performance compared to other models due to its ability to capture temporal patterns and dependencies in the stock price time series.

The repository contains Jupyter notebooks that showcase our approach in detail. These notebooks cover data analysis, preprocessing, model development (linear regression, ensemble learning, LSTM stacking), hyperparameter tuning, and visualization of results.

## Conclusion:
The ML Hackrush-23 challenge provided an exciting opportunity for our team, predict.me(), to apply AI and advanced algorithms for stock market prediction. By combining techniques like linear regression with hyperparameter tuning, ensemble learning, dimensionality reduction using PCA, and LSTM stacking, we achieved top results in forecasting the stock prices of TGD Consultancy, TGD Automobiles, and TGD Power.

In the future, we aim to explore advanced techniques such as Transformers for improved modeling and uncertainty estimation techniques like dropout and Bayesian inference. These advancements will help us further enhance the accuracy and reliability of our predictions, enabling better investment decisions and maximizing profits in the stock market.

#### That's it!!

ML Challenge Winner: Team Name - predict.me() | Leaderboard Position: 1 | 36 hours of relentless teamwork, fueled by subways and caffeine, resulted in securing the top position.

Please find the Kaggle Competition here:-  [here](https://www.kaggle.com/competitions/forecasting-stocks-using-aiml/overview)
