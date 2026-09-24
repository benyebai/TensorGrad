import numpy as np
import matplotlib.pyplot as plt
from tensorgrad import Tensor

rng = np.random.default_rng(42)

# 32 examples, each containing x1 and x2.
inputs = rng.uniform(-5.0, 5.0, size=(32, 2))

# Hidden rule: y = 2*x1 - 3*x2 + 1
targets = (
    2.0 * inputs[:, 0]
    - 3.0 * inputs[:, 1]
    + 1.0
).reshape(32, 1)

inputs_tensor = Tensor(inputs)
targets_tensor = Tensor(targets)

weights = Tensor(rng.uniform(-1.0, 1.0, size=(2, 1)))
bias = Tensor(np.zeros(1)) # i gotta review how broadcasting and numpy work, but mainly it aligns from the right

learning_rate = 0.01
losses = []

for step in range(700):
    predictions = inputs_tensor @ weights + bias # shape (32,1)
    loss = ((predictions - targets_tensor)**2).mean()
    losses.append(float(loss.data))
    weights.grad = np.zeros_like(weights.data)
    bias.grad = np.zeros_like(bias.data)
    loss.backward()
    weights.data -= learning_rate * weights.grad
    bias.data -= learning_rate * bias.grad

print(f"final loss: {losses[-1]:.3e}")
print(f"weights: {weights.data.ravel()}")
print(f"bias: {bias.data.item():.6f}")

plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.yscale("log")
plt.title("TensorGrad Linear Regression Training")
plt.xlabel("Training Step")
plt.ylabel("Mean Squared Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("linear_regression_training.png", dpi=200)
print("saved graph: linear_regression_training.png")
