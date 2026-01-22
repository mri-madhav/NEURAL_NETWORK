
---

## 🔢 Mathematical Details

### Activation Functions
- ReLU:  f(z) = max(0, z)
- Softmax: Converts logits into class probabilities

### Loss Function
Categorical Cross-Entropy:
L = - (1 / m) ∑ y log(ŷ)

Using Softmax + Cross-Entropy gives the simplified gradient:
∂L / ∂Z = ŷ − y

---

## ⚙️ Implementation Details
- Manual implementation of forward propagation and backpropagation
- He initialization for stable training with ReLU
- Numerically stable softmax and loss
- Vectorized NumPy operations
- Reproducible results using fixed random seed

---

## 📊 Training Setup

| Parameter | Value |
|----------|-------|
| Optimizer | Gradient Descent |
| Learning Rate | 0.01 |
| Epochs | 1000 |
| Initialization | He Initialization |
| Batch Type | Full Batch |

---

## ✅ Results

| Dataset | Accuracy |
|--------|----------|
| Training | ~97–98% |
| Test | ~95–96% |

---

## 📁 Project Structure
