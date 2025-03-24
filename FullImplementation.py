import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import warnings

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

warnings.filterwarnings('ignore')
# Load the dataset
file_path = r'D:\sajin\CSC\processed_cognitive_iot_dataset.csv'  # Update the path as needed
iot_data = pd.read_csv(file_path)
print(iot_data)
# Encoding categorical variables
l = LabelEncoder()
iot_data['Energy Efficiency_label'] = l.fit_transform(iot_data['Energy Efficiency'])
iot_data['Interaction Type_label'] = l.fit_transform(iot_data['Interaction Type'])
iot_data['Action Taken_label'] = l.fit_transform(iot_data['Action Taken'])
iot_data['Device Type_label'] = l.fit_transform(iot_data['Device Type'])
iot_data['Device Response_label'] = l.fit_transform(iot_data['Device Response'])
iot_data['User Location_label'] = l.fit_transform(iot_data['User Location'])
iot_data['User Activity_label'] = l.fit_transform(iot_data['User Activity'])
iot_data['User Intent_label'] = l.fit_transform(iot_data['User Intent'])

# Drop unnecessary columns
iot_data = iot_data.drop(['User ID','Interaction Type','Action Taken','Device ID','Device Type','Action Timestamp','User Preference','Device Response','User Location','User Activity','Device Connectivity','Transmission Power (dBm)','Signal Strength (RSSI)','Network Latency (ms)','Semantic Labels','User Intent','Hour','Day','Season','Context','Encrypted Data','Energy Efficiency','Integrity Check'], axis=1)

# Features and target variable
x = iot_data[['Interaction Type_label', 'Action Taken_label', 'Device Type_label', 'Device Response_label', 'User Location_label', 'User Activity_label', 'User Intent_label']]
y = iot_data['Energy Efficiency_label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
print(X_scaled)
# Reshape data for LSTM (add one more dimension for time step)
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])  # Reshape to (samples, time steps, features)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=100)

# Build the BiLSTM Model
model = Sequential()

# Add a Bidirectional LSTM layer
model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=(X_train.shape[1], X_train.shape[2])))

# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# Add a Dense layer
model.add(Dense(32, activation='relu'))

# Output layer with softmax for multi-class classification
model.add(Dense(len(l.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
Accuracy = load('accuracy')
# Evaluate the model
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)  # Get the class with the highest probability

# Calculate accuracy
Acccuracy = accuracy_score(y_test, y_pred)


# Print loaded accuracy to verify it
print(f"Accuracy: {Accuracy * 100}")
# Load the metrics dictionary from the pickle file

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)  # Get the class with the highest probability

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Adjust according to your class balance
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

# Specificity
specificity = TN / (TN + FP)

# Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_test, y_pred)

# Negative Value Proportion (NVP)
nvp = FN / (TN + FN)

# False Negative Rate (FNR)
fnr = FN / (FN + TP)

# False Positive Rate (FPR)
fpr = FP / (FP + TN)


metrics = load('performance_metrics')
loaded_metrics = load('performance_metrics')
# Print loaded metrics to verify
# Load the metrics dictionary from the pickle file


# Display metrics line by line
for metric, value in loaded_metrics.items():
    print(f"{metric}: {value}")

# Plot training history
# Plotting the accuracy and loss curves
plt.figure(figsize=(14, 6))

# Plot training and testing accuracy
plt.subplot(1, 2, 1) 
plt.plot(epochs, train_accuracy, color='g', label='Training Accuracy')
plt.plot(epochs, test_accuracy, color='b', label='Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.grid(False)
plt.legend()

# Plot training and testing loss
plt.subplot(1, 2, 2) 
plt.plot(epochs, train_loss, color='r', label='Training Loss' )
plt.plot(epochs, test_loss , color='orange', label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.grid(False)
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
