# this is a tutorial for understanding pytorch
import matplotlib.pyplot as plt
import torch

# let's say we have input x
x = torch.Tensor([2])
# a parameter w we want to optimize
w = torch.Tensor([3])
# w is a parameter we want to optimize, so we want to it to keep track of its gradient
w.requires_grad = True

# once you have w.requires_grad = True, you have two properties of w
# w.data is the actual value of w
print(f"w.data = {w.data}")
# w.grad is the gradient of w
print(f"w.grad = {w.grad}")
# it is None because we haven't done any backward pass yet

# say the output is y
y = x*w

# we can already calculate dy/dw
y.backward()

# dy/dw should be x, so let's print it
print(f"w.grad after backward = {w.grad}")

# say you want to do gradient descent on w
# you want to do w = w - lr * dw/dw
# you can do
w.data = w.data - 0.01 * w.grad

# it is for now no clear if we have successfully done optimize any objective
# so let's make the objective more clear

w.grad.zero_()

# let's reset w first
w = torch.Tensor([3])
w.requires_grad = True
# let's say we want to minimize the loss
y = x*w
loss = (y - 4).pow(2)*0.5

# we can do backward on loss
loss.backward()

# now we can see the gradient of w is not x anymore
# it should be d loss / d w = d loss / d y * d y / d w = (y - 4) * x = (2 * 3 - 4) * 2 = 4
# let's see if we can get the same result
print(f"w.grad after backward = {w.grad}")

# now you can take steps to optimize w
w.data = w.data - 0.05 * w.grad

# let's put that in a loop so that we can see the optimization progress

# let's hold a list of loss and w to plot later

ws = []
losses = []

for epoch in range(20):
    y = x*w
    loss = (y - 4).pow(2)*0.5

    # before backward, we need to zero out the gradient
    w.grad.zero_()
    # if you don't zero out the gradient, the gradient will accumulate

    loss.backward()

    w.data = w.data - 0.05 * w.grad

    print(f"epoch {epoch}: w = {w.data}, loss = {loss.data}")

    # let's save the w and loss for plotting
    ws.append(w.data.item())
    losses.append(loss.data.item())

# let's plot loss and w
plt.plot(losses)
plt.show()

plt.plot(ws)
plt.show()
