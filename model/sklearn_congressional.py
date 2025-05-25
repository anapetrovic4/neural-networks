from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable
import pandas as pd

def count_parameters(model):
    total_params = 0
    for i, coef in enumerate(model.coefs_):
        layer_params = coef.shape[0] * coef.shape[1]
        print(f"Layer {i} weights: {coef.shape} -> {layer_params} params")
        total_params += layer_params

    for i, bias in enumerate(model.intercepts_):
        bias_params = len(bias)
        print(f"Layer {i} biases: {bias.shape} -> {bias_params} params")
        total_params += bias_params

    print(f"Total trainable parameters: {total_params}")
    return total_params

df = pd.read_csv('../datasets/clean_congressional.csv')
X = df.drop(columns=['class'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), max_iter=1000, random_state=42)

param_grid = {
        'hidden_layer_sizes': [(4,), (8,), (16,)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [30, 50],
        'batch_size': [16, 32]
    }

grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    cv=2,
    scoring='accuracy',
    verbose=2
)

grid_search.fit(X_train, y_train)

# Results
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
test_acc = best_model.score(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

#y_pred = mlp.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy*100:.2f}%")

# Count parameters
count_parameters(best_model)