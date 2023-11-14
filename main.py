import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


# Define the RNN model
def build_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(units=64, input_shape=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Load and preprocess primary cohort
primary_cohort_path = 's41598-020-73558-3_sepsis_survival_primary_cohort.csv'
scaler = StandardScaler()
X_primary, y_primary = preprocess_data(primary_cohort_path, scaler, fit_scaler=True)

# Split primary cohort data
X_train_primary, X_test_primary, y_train_primary, y_test_primary = train_test_split(X_primary, y_primary, test_size=0.2,
                                                                                    random_state=42)

# Build and train model on primary cohort
model = build_model((1, X_train_primary.shape[2]))
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
Training on Primary Cohort:

Loss and Accuracy During Training: The loss is relatively low, and the accuracy is high (around 92.59%) throughout the training. However, the accuracy doesn't seem to improve much across epochs. This could indicate that the model has quickly reached its performance limit given the current architecture and data, or it could suggest that the data is imbalanced (if most records belong to one class).
Validation Performance: Similar accuracy on the validation set (around 93.05%) indicates that the model is not overfitting to the training data.
Performance on Study Cohort:

Accuracy: The model achieved around 81.07% accuracy on the study cohort. This is lower than the training accuracy, which is expected as the model is now being tested on a new set of data that it hasn't seen before.
Performance on Validation Cohort:

Accuracy: The accuracy is about 82.48% on the validation cohort. This cohort, being from South Korea, represents a different demographic, so it's encouraging to see the model still performs reasonably well.
Key Takeaways and Considerations:
Generalizability: The model is somewhat generalizable, as indicated by its performance on different cohorts. However, the drop in accuracy from the training to the study/validation cohorts suggests there may be room for improvement.
Data Imbalance: If the primary cohort data is imbalanced, it might have affected the model's ability to learn diverse patterns. This might explain the high accuracy during training (the model might be predicting the majority class well) and the lower accuracy on other cohorts.
Model Complexity and Tuning: The RNN architecture used is relatively simple. Experimenting with more complex architectures, hyperparameter tuning, or different types of layers (like LSTM or GRU) could potentially improve performance.
Data Representation: RNNs can be sensitive to how data is presented. Since the data is not sequential (like text or time series), other model types (like feedforward neural networks or ensemble methods) might be more suitable.
Evaluation Metrics: Consider using additional metrics like precision, recall, F1-score, and confusion matrices, especially if the dataset is imbalanced. These metrics can provide a more nuanced understanding of my model's performance.
In summary, the model shows promise but may benefit from further refinement and exploration of different modeling techniques and evaluation strategies.





'''