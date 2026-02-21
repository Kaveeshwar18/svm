import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='rbf', gamma='scale', probability=True)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model saved successfully as model.pkl")