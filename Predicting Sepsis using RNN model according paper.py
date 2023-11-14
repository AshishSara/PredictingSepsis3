'''
Here's a brief overview:

Preprocessing Function: The preprocess_data function correctly reads the data, selects the necessary features, applies normalization using MinMaxScaler, and reshapes the data for RNN input.

Model Definition: The build_gru_model function sets up a GRU-based sequential model with the specified architecture, including two GRU layers with 40 units each, followed by a dense output layer with sigmoid activation (suitable for binary classification).

Data Preparation for Primary Cohort: The primary cohort data is loaded, preprocessed, and split into training and testing sets.

Model Training: The model is compiled and trained on the primary cohort training data, with a validation split to monitor performance during training.

Evaluation on Study and Validation Cohorts: After training, the model is evaluated on both the study cohort and the validation cohort, which provides insight into its generalizability and performance on different data sets.


'''



import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # Changed to MinMaxScaler for normalization
from sklearn.metrics import roc_auc_score


# Function to preprocess and reshape data
# Function to preprocess and reshape data
def preprocess_data(filepath, scaler=None, fit_scaler=False):
    data = pd.read_csv(filepath)
    X = data[['age_years', 'sex_0male_1female', 'episode_number']]
    y = data['hospital_outcome_1alive_0dead']

    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    X_rnn = X.reshape((X.shape[0], 1, X.shape[1]))
    return X_rnn, y



# Define the RNN model with GRU
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(40, return_sequences=True, input_shape=input_shape, activation='relu'))  # First GRU layer
    model.add(GRU(40, activation='relu'))  # Second GRU layer
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Load and preprocess primary cohort
primary_cohort_path = 's41598-020-73558-3_sepsis_survival_primary_cohort.csv'
scaler = MinMaxScaler()  # Use MinMaxScaler here
X_primary, y_primary = preprocess_data(primary_cohort_path, scaler, fit_scaler=True)

# Split primary cohort data
X_train_primary, X_test_primary, y_train_primary, y_test_primary = train_test_split(X_primary, y_primary, test_size=0.2, random_state=42)


# Build and train the model on primary cohort
model = build_gru_model((1, X_train_primary.shape[2]))  # Correct shape reference
history = model.fit(X_train_primary, y_train_primary, epochs=10, batch_size=32, validation_split=0.1)

# Load, preprocess, and evaluate on study cohort
study_cohort_path = 's41598-020-73558-3_sepsis_survival_study_cohort.csv'
X_study, y_study = preprocess_data(study_cohort_path, scaler)
loss_study, accuracy_study = model.evaluate(X_study, y_study)
print(f'Study Cohort Test accuracy: {accuracy_study:.4f}')

# Load, preprocess, and evaluate on validation cohort
validation_cohort_path = 's41598-020-73558-3_sepsis_survival_validation_cohort.csv'
X_validation, y_validation = preprocess_data(validation_cohort_path, scaler)
loss_validation, accuracy_validation = model.evaluate(X_validation, y_validation)
print(f'Validation Cohort Test accuracy: {accuracy_validation:.4f}')

'''
results:
/Users/ashishsaragadam/anaconda3/bin/python /Users/ashishsaragadam/Desktop/Python/KaggleSepsis/Predicting Sepsis using RNN model according paper.py 
WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
Epoch 1/10
2480/2480 [==============================] - 5s 1ms/step - loss: 0.2594 - accuracy: 0.9249 - val_loss: 0.2319 - val_accuracy: 0.9305
Epoch 2/10
2480/2480 [==============================] - 3s 1ms/step - loss: 0.2452 - accuracy: 0.9259 - val_loss: 0.2319 - val_accuracy: 0.9305
Epoch 3/10
2480/2480 [==============================] - 3s 1ms/step - loss: 0.2450 - accuracy: 0.9259 - val_loss: 0.2329 - val_accuracy: 0.9305
Epoch 4/10
2480/2480 [==============================] - 4s 2ms/step - loss: 0.2450 - accuracy: 0.9259 - val_loss: 0.2323 - val_accuracy: 0.9305
Epoch 5/10
2480/2480 [==============================] - 4s 2ms/step - loss: 0.2448 - accuracy: 0.9259 - val_loss: 0.2316 - val_accuracy: 0.9305
Epoch 6/10
2480/2480 [==============================] - 3s 1ms/step - loss: 0.2448 - accuracy: 0.9259 - val_loss: 0.2329 - val_accuracy: 0.9305
Epoch 7/10
2480/2480 [==============================] - 4s 1ms/step - loss: 0.2447 - accuracy: 0.9259 - val_loss: 0.2315 - val_accuracy: 0.9305
Epoch 8/10
2480/2480 [==============================] - 3s 1ms/step - loss: 0.2446 - accuracy: 0.9259 - val_loss: 0.2335 - val_accuracy: 0.9305
Epoch 9/10
2480/2480 [==============================] - 4s 1ms/step - loss: 0.2445 - accuracy: 0.9259 - val_loss: 0.2321 - val_accuracy: 0.9305
Epoch 10/10
2480/2480 [==============================] - 4s 1ms/step - loss: 0.2445 - accuracy: 0.9259 - val_loss: 0.2315 - val_accuracy: 0.9305
596/596 [==============================] - 0s 580us/step - loss: 0.5317 - accuracy: 0.8107
Study Cohort Test accuracy: 0.8107
5/5 [==============================] - 0s 871us/step - loss: 0.5665 - accuracy: 0.8248
Validation Cohort Test accuracy: 0.8248

Process finished with exit code 0

Interpretation:
Training Performance:

The model achieves a consistent training accuracy of around 92.59% across all epochs. The loss decreases slightly, starting at 0.2594 and reducing to 0.2445 by the end of training.
The validation accuracy remains constant at approximately 93.05%, which is a good sign of the model's stability. However, the lack of improvement in validation loss and accuracy over epochs may indicate that the model has reached its performance limit with the current configuration.
Performance on Study Cohort:

The accuracy on the study cohort is 81.07%. This drop in accuracy compared to the training performance suggests that the model may not generalize as well to this new dataset. However, it still maintains a reasonably high accuracy.
Performance on Validation Cohort:

The model achieves an accuracy of 82.48% on the validation cohort. This further demonstrates the model's ability to generalize, albeit with some loss in accuracy compared to the training set.
'''

# Predict probabilities on the study cohort
y_pred_study = model.predict(X_study).ravel()  # ravel() flattens the array
# Compute AUROC for the study cohort
auroc_study = roc_auc_score(y_study, y_pred_study)
print(f'Study Cohort AUROC: {auroc_study:.4f}')

# Predict probabilities on the validation cohort
y_pred_validation = model.predict(X_validation).ravel()
# Compute AUROC for the validation cohort
auroc_validation = roc_auc_score(y_validation, y_pred_validation)
print(f'Validation Cohort AUROC: {auroc_validation:.4f}')

'''
596/596 [==============================] - 0s 493us/step
Study Cohort AUROC: 0.5887
5/5 [==============================] - 0s 640us/step
Validation Cohort AUROC: 0.5706

AUROC Score Interpretation:

An AUROC score ranges from 0 to 1, where a score of 0.5 indicates no discriminative ability (equivalent to random guessing), and a score of 1 indicates perfect discrimination between the positive and negative classes.
My scores of 0.5887 for the study cohort and 0.5706 for the validation cohort suggest that the model has limited ability to distinguish between the two classes (survived or deceased). These scores are only slightly better than random guessing.
Model Performance:

While the accuracy of the model was relatively high, the AUROC scores indicate that the model might not be effectively differentiating between the classes, especially in more balanced scenarios.
This could be due to several factors, such as class imbalance, inadequate model complexity, or insufficiently informative features.
Next Steps:

Address Imbalance: If the dataset is imbalanced, consider techniques like oversampling the minority class, undersampling the majority class, or using class weights.
Model Architecture: Experiment with different model architectures or more complex models. GRU units are powerful, but the model might require additional layers or a different setup to capture the complexities of the data.
Feature Engineering: Investigate if additional features can be included or if existing features can be transformed to better capture the patterns in the data.
Hyperparameter Tuning: Fine-tuning the hyperparameters of the model (like learning rate, number of units in GRU layers, etc.) might improve performance.
Clinical Relevance:

Given the serious nature of sepsis, it's crucial to develop a highly accurate and reliable predictive model. Collaborate with medical professionals to understand the nuances of the data and to ensure that the model's predictions are clinically relevant.

'''