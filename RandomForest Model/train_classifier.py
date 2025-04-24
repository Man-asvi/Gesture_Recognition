import pickle
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
try:
    with open('./data.pickle', 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: 'data.pickle' file not found. Make sure it exists in the current directory.")
    exit()

# Convert to numpy arrays
data = np.asarray(data_dict.get('data', []))
labels = np.asarray(data_dict.get('labels', []))

# Print dataset info
print(f"✅ Data shape: {data.shape}")
print(f"✅ Labels shape: {labels.shape}")

# Check if data is valid
if len(data) == 0 or len(labels) == 0:
    print("❌ Error: The dataset is empty. Check your data collection or feature extraction process.")
    exit()

# Show class distribution
class_counts = Counter(labels)
print("✅ Class distribution:", class_counts)

# Make sure all classes have at least 2 samples for stratified split
if any(count < 2 for count in class_counts.values()):
    print("❌ Error: At least one class has fewer than 2 samples. Cannot perform stratified split.")
    exit()

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"✅ Accuracy: {score * 100:.2f}% of samples classified correctly!")

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("✅ Model saved to 'model.p'")
