<<<<<<< HEAD
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

iris_data = pd.read_csv('dataset/Iris.csv')

iris_data['Species'] = iris_data['Species'].astype('category')

# Splitting the data into features and target
X = iris_data.drop(columns=['Species'])
y = iris_data['Species']

# Train-test split (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
print(f"Logistic Regression Accuracy: {log_reg_acc:.2f}")
print("\nClassification Report - Logistic Regression:\n", classification_report(y_test, log_reg_pred))

# Model 2: Support Vector Machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_acc:.2f}")
print("\nClassification Report - SVM:\n", classification_report(y_test, svm_pred))

# Model 3: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.2f}")
print("\nClassification Report - Random Forest:\n", classification_report(y_test, rf_pred))

# Data visualization - Scatter plots for each pair of features
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_data, x='SepalLengthCm', y='SepalWidthCm', hue='Species', palette='Set2')
plt.title('Sepal Length vs Sepal Width')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_data, x='PetalLengthCm', y='PetalWidthCm', hue='Species', palette='Set2')
plt.title('Petal Length vs Petal Width')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_data, x='SepalLengthCm', y='PetalLengthCm', hue='Species', palette='Set2')
plt.title('Sepal Length vs Petal Length')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_data, x='SepalWidthCm', y='PetalWidthCm', hue='Species', palette='Set2')
plt.title('Sepal Width vs Petal Width')
=======
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

iris_data = pd.read_csv('dataset/Iris.csv')

iris_data['Species'] = iris_data['Species'].astype('category')

# Splitting the data into features and target
X = iris_data.drop(columns=['Species'])
y = iris_data['Species']

# Train-test split (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
print(f"Logistic Regression Accuracy: {log_reg_acc:.2f}")
print("\nClassification Report - Logistic Regression:\n", classification_report(y_test, log_reg_pred))

# Model 2: Support Vector Machine (SVM)
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_acc:.2f}")
print("\nClassification Report - SVM:\n", classification_report(y_test, svm_pred))

# Model 3: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.2f}")
print("\nClassification Report - Random Forest:\n", classification_report(y_test, rf_pred))

# Data visualization - Scatter plots for each pair of features
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_data, x='SepalLengthCm', y='SepalWidthCm', hue='Species', palette='Set2')
plt.title('Sepal Length vs Sepal Width')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_data, x='PetalLengthCm', y='PetalWidthCm', hue='Species', palette='Set2')
plt.title('Petal Length vs Petal Width')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_data, x='SepalLengthCm', y='PetalLengthCm', hue='Species', palette='Set2')
plt.title('Sepal Length vs Petal Length')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris_data, x='SepalWidthCm', y='PetalWidthCm', hue='Species', palette='Set2')
plt.title('Sepal Width vs Petal Width')
>>>>>>> 54dd8901e61f0df9711c22cb12dc11d9b3fb8a0d
plt.show()