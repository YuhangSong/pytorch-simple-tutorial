# this is a tutorial for understanding pytorch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# let's reset w first
w = torch.Tensor([3])
w.requires_grad = True

for epoch in range(20):
    y = x*w
    loss = (y - 4).pow(2)*0.5

    loss.backward()

    w.data = w.data - 0.05 * w.grad

    print(f"epoch {epoch}: w = {w.data}, loss = {loss.data}")

    # let's save the w and loss for plotting
    ws.append(w.data.item())
    losses.append(loss.data.item())

    # before going forward to the next loop, we need to zero out the gradient
    w.grad.zero_()
    # if you don't zero out the gradient, the gradient will accumulate

# let's plot loss and w
plt.plot(losses)
plt.show()

plt.plot(ws)
plt.show()

# so now you see, you can really code up a very complex process of information processing and define a loss function in the end
# and optmized it with minimal effort

# this is the whole core concept of auto-differentiation

# pytorch makes the above process even more packed, so for now you still manually do gradient descent update on w: w.data = w.data - 0.05 * w.grad
# this can be packed into a optimizer

# let's reset w first
w = torch.Tensor([3])
w.requires_grad = True

# let's create an optimizer
optimizer = torch.optim.SGD([w], lr=0.05)

# let's put that in a loop so that we can see the optimization progress
for epoch in range(20):
    y = x*w
    loss = (y - 4).pow(2)*0.5

    # before backward, we need to zero out the gradient
    # instead of do w.grad.zero_() for all the parameters, we can do optimizer.zero_grad() which will do it for all the parameters
    optimizer.zero_grad()
    # if you don't zero out the gradient, the gradient will accumulate

    loss.backward()

    # instead of do w.data = w.data - 0.05 * w.grad
    # now you can do optimizer step
    optimizer.step()

    print(f"epoch {epoch}: w = {w.data}, loss = {loss.data}")

    # let's save the w and loss for plotting
    ws.append(w.data.item())
    losses.append(loss.data.item())

# let's plot loss and w
# I intentionally keep the list uncleaned so that you can see there are no difference between the two
# but just easier to code with optimizer
plt.plot(losses)
plt.show()

plt.plot(ws)
plt.show()

# additionally, there are many options for optimizer than just SGD (gradient descent), which you can use with minimal effort

# so far you should say pytorch (auto-differentiation) is not just for training neural networks, the concept of auto-differentiation is very general
# you can use it for other optimization purposes

# a comment here, I am very attached to a concept: simple thing works best (contrary to the common practice in research)
# first order gradient is the simplest thing that it works in a wide range of applications, you can imagine there are many
# second order optimization methods, which might not work as widely as first order optimization

# now let's go back to neural networks

# so training neural networks is nothing more than just make the w we were usign a complex network

# let's say we have a simple network we want to optimize (replacing the single w we optmized before)
model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

# then let's create a xor dataset
data = torch.Tensor([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
])

target = torch.Tensor([
    [1],
    [-1],
    [-1],
    [1],
])

# let's create a optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

losses = []

# let's put that in a loop so that we can see the optimization progress
for epoch in range(100):

    y = model(data)
    loss = (y - target).pow(2).sum()*0.5

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(f"epoch {epoch}: loss = {loss.data}")
    losses.append(loss.data.item())

# let's plot loss
plt.plot(losses)
plt.show()
