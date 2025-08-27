import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

iris = load_iris()
indata = pd.DataFrame(data = iris.data , columns = iris.feature_names)
print(indata.head())
print(iris.target)
indata['species'] = iris.target_names[iris.target]
print(indata.head())
print(indata['species'].value_counts())
# indata.to_excel('iris_data.xlsx', index=False)

# Scatter plot:
plt.figure(figsize=(8, 6))
sns.scatterplot(data=indata, x='petal length (cm)', y='petal width (cm)', hue='species', style='species')
plt.title('Petal Length vs Petal Width by Species', fontsize=14)
plt.xlabel('Petal Length (cm)', fontsize=12)
plt.ylabel('Petal Width (cm)', fontsize=12)
plt.grid(True)
plt.savefig('iris_petal_scatter.png', dpi=300)
plt.show()
# Creat and test model:
X = indata.drop('species', axis=1)
y = indata['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Save model and prediction:
joblib.dump(model, 'iris_model.pkl')
sample = [[5.0, 3.4, 1.5, 0.2]]
predicted_species = model.predict(sample)[0]
print(f"Predicted species: {predicted_species}")