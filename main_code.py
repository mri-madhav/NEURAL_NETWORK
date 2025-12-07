import numpy as np
def linear(z):
    return z
def relu(z):
     return np.maximum(0, z)
def softmax(z):
    ez=np.exp(z)
    sum=np.sum(ez)
    return (ez/sum)
def sigmoid(z):
    return(1/(1+np.exp(-z)))

n=int(input("Number of features="))
arr = list(map(int, input("Enter array: ").split()))
#layer size
layer_sizes = [n, 64, 32, 4]

def parameters(layer_sizes):
    param={}
    L=len(layer_sizes)
    for i in range (1,L):
        param[f"w{i}"]=np.random.rand(layer_sizes[i], layer_sizes[i-1])
        param[f"b{i}"]=np.zeros((layer_sizes[i], 1))
    return param
#defineing backpropagation
def back_propagation(param, X):
    L=len(layer_sizes)-1
    A=X
    caches={}
    for i in range (1,L):
        
        W=param[f"W{i}"]
        b=param[f"b{i}"]
        z=np.dot(W,A)+b
        A=relu(z)
        caches[f"A{i}"] = A
        caches[f"Z{i}"] = z
    #it is the final layer
    W = param[f"w{L}"]
    b = param[f"b{L}"]
    Z = np.dot(W, A) + b
    A = softmax(Z)
    caches[f"A{L}"]=A
    caches[f"Z{L}"]=Z
    return A, caches

