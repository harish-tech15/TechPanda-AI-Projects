# AttriPredict AI – Integrated ANN

An end-to-end employee attrition prediction implementation that combines data preprocessing and ANN prediction into one integrated pipeline.

## Workflow
1. Load the IBM HR Employee Attrition dataset.
2. Separate features and target.
3. Build a preprocessing pipeline using ColumnTransformer.
4. Scale numerical features.
5. Encode categorical features.
6. Train the ANN model.
7. Evaluate model performance.
8. Test predictions using the integrated pipeline.

## Model
- Input features after preprocessing: 51
- Hidden layers: 64 and 32 neurons
- Output layer: 1 neuron with sigmoid activation
- Framework: TensorFlow / Keras

## Test Results
- Accuracy: 80.95%
- ROC-AUC: 0.7872
- Leave (Yes) Recall: 55%

## Saved Artifacts
- Integrated ANN model
- Integrated preprocessing pipeline

## Project Status
Completed
