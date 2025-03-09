# F1 Podium Prediction Models

This directory contains machine learning models for predicting Formula 1 race podiums.

## Models

1. **XGBoost Model** (`xg-boost.py`): A gradient boosting model for podium prediction.
2. **Neural Network Model** (`neural_network.py`): A feed-forward neural network for podium prediction using PyTorch.
3. **Grid Search** (`grid_search.py`): A script to find optimal hyperparameters for the neural network model.
4. **Optimized Neural Network** (`optimized_nn.py`): A neural network model using the best hyperparameters found by grid search.

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Make sure the dataset is available at `../cleaned/f1_podium_prediction_dataset.csv`

## Running the Models

To run the XGBoost model:
```
python xg-boost.py
```

To run the basic Neural Network model:
```
python neural_network.py
```

To perform hyperparameter optimization with grid search:
```
python grid_search.py
```

To train and evaluate the optimized neural network model:
```
python optimized_nn.py
```
Note: The optimized model requires grid search results. Run `grid_search.py` first.

## Hyperparameter Optimization

The grid search explores different combinations of:
- Network architecture (hidden layer sizes)
- Dropout rates
- Learning rates
- Batch sizes
- Weight decay (L2 regularization)

Results are saved to a JSON file that can be used by the optimized model.

## Model Features

All models use the following features:
- Driver information (career stats, previous season performance)
- Constructor information
- Circuit information
- Recent performance metrics
- Grid position

## Evaluation Metrics

The models are evaluated using several custom metrics:
- Position-specific accuracy (1st, 2nd, 3rd place)
- Podium inclusion accuracy
- Mean positional distance error
- Complete podium accuracy
- Weighted podium score

Standard classification metrics are also provided for reference. 