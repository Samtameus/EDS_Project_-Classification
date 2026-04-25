from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data 
Y = iris.target

## print("Features:\n", X[:5].tolist())  
## print("Labels:\n", Y[:5].tolist())    
## print("Target Names: \n", iris.target_names.tolist()) 

#Splitting the dataset into training and testing sets.

X_train = np.vstack([X[0:30], X[50:80], X[100:130]]) #consist of 30 samples (flower and 4 features )from each class
Y_train = np.hstack([Y[0:30], Y[50:80], Y[100:130]]) #labels (type of flower)

X_test = np.vstack([X[30:50], X[80:100], X[130:150]])
Y_test = np.hstack([Y[30:50], Y[80:100], Y[130:150]])

## print("\nX_train shape:", X_train.shape)  # Should be (90, 4)
## print("Y_train shape:", Y_train.shape)  # Should be (90,)
## print("X_test shape:", X_test.shape)   # Should be (60, 4)
## print("Y_test shape:", Y_test.shape)   # Should be (60,)

## print("\nTraining labels:", Y_train)# should contain 30 samples of each class (0, 1, 2)
#Helping functions 

def add_bias(X):
    N = X.shape[0]
    bias = np.ones((N, 1))  # Create a column of ones for
    return np.hstack([X, bias])


def one_hot(labels, num_classes = 3):
    N = len(labels)
    T = np.zeros((N, num_classes))
    for i, label in enumerate(labels): 
        T[i, label] = 1
    return T                


# applying the functions to the training and testing data
X_train_b = add_bias(X_train)
X_test_b = add_bias(X_test)

T_train = one_hot(Y_train)

print("X_train_b shape:", X_train_b.shape)  # Should be (90, 5)
print("T_train shape:  ", T_train.shape)    # Should be (90, 3)

# Clasifier

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def forward(X_b, W): 
    z = X_b @ W.T # matrix multiplication
    return sigmoid(z)


def compute_mse(g, T):
    error = g - T
    mse = 0.5 * np.mean(np.sum(error**2, axis=1))
    return mse  


def compute_gradient(X_b, g, T):
    error = g - T                  # how wrong per output
    delta = error * g * (1 - g)  # element-wise multiplication
    return delta.T @ X_b

# ---------- gå gjennom fra her ----------
def train(X_b, T, alpha=0.01, epochs=1000):
    """
    Equation 23 - the full training loop
    alpha  = step size (how big each adjustment is)
    epochs = how many times we go through the data
    """
    C = T.shape[1]        # 3 classes
    D = X_b.shape[1]      # 5 inputs

    W = np.zeros((C, D))  # start with all zeros
    mse_history = []

    for epoch in range(epochs):
        g    = forward(X_b, W)           # what does W predict?
        mse  = compute_mse(g, T)         # how wrong is it?
        grad = compute_gradient(X_b, g, T)  # which way to move?
        W    = W - alpha * grad          # Equation 23 - take a step

        mse_history.append(mse)

    return W, mse_history


# train the model and get the weights and the history of MSE

W, mse_history = train(X_train_b, T_train, alpha=0.01, epochs=1000)

# Plot MSE - this is how you check if training worked
plt.plot(mse_history)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training convergence')
plt.grid(True)
plt.show()

print("Starting MSE:", round(mse_history[0], 4))
print("Final MSE:   ", round(mse_history[-1], 4))
