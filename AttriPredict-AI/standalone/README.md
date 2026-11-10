# AttriPredict AI – Standalone ANN

This implementation independently performs data preprocessing and employee attrition prediction using an Artificial Neural Network.

## Workflow
1. Load the IBM HR Employee Attrition dataset.
2. Clean and prepare the data.
3. Encode categorical features.
4. Scale numerical features.
5. Train the ANN model.
6. Evaluate model performance.
7. Test employee attrition predictions.

## Model
- Input layer: 51 features
- Hidden layers: 64 and 32 neurons
- Output layer: 1 neuron with sigmoid activation
- Framework: TensorFlow / Keras

## Test Results
- Accuracy: 82.65%
- ROC-AUC: 0.7905

## Artifacts
- ANN model
- Encoder
- Scaler
- Feature column information

## Project Status
Completed
