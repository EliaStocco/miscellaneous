import torch

def func_1(x):
    # it does NOT work
    return torch.tensor([x[0] * x[1], x[1] + 2],requires_grad=True)

def func_2(x):
    # it DOES work
    y = torch.empty_like(x)
    y[0] = x[0] * x[1]
    y[1] = x[1] + 2
    return y

def func_3(x):
    # it DOES work
    y0 = x[0] * x[1]
    y1 = x[1] + 2
    return torch.hstack((y0,y1))

# Input tensor
x = torch.tensor([5.0, 7.0], requires_grad=True)

# Using torch.autograd.functional.jacobian
print("\nusing torch.autograd.functional.jacobian")
jacobian = torch.autograd.functional.jacobian(func_1, x)
print("Jacobian Matrix (1):\n",jacobian,"\n")

jacobian = torch.autograd.functional.jacobian(func_2, x)
print("Jacobian Matrix (2):\n",jacobian,"\n")

# it works with both 'hstack' and 'vstack'
jacobian = torch.autograd.functional.jacobian(func_3, x)
print("Jacobian Matrix (3):\n",jacobian,"\n")

# Using torch.backward
print("\nusing torch.backward")
y = func_1(x)
y[0].backward()
print("Gradient of y[0] (1):", x.grad)
y = func_1(x)
y[1].backward()
print("Gradient of y[1] (1):", x.grad)

y = func_2(x)
y[0].backward()
print("Gradient of y[0] (2):", x.grad)
y = func_1(x)
y[1].backward()
print("Gradient of y[1] (2):", x.grad)

# it works with both 'hstack' and 'vstack'
y = func_3(x)
y[0].backward()
print("Gradient of y[0] (3):", x.grad)
y = func_1(x)
y[1].backward()
print("Gradient of y[1] (3):", x.grad)

# Using torch.func.grad
print("\nusing torch.func.grad")
try :
    # this should NOT work
    jacobian = torch.func.grad(lambda x:func_1(x)[0])
    print("Gradient of y[0] (1):\n",jacobian(x),"\n")
    jacobian = torch.func.grad(lambda x:func_1(x)[1])
    print("Gradient of y[1] (1):\n",jacobian(x),"\n")
except:
    print("Gradient of y[0] (1):\n",None,"\n")
    print("Gradient of y[1] (1):\n",None,"\n")

try :
    # this should DO work
    jacobian = torch.func.grad(lambda x:func_2(x)[0])
    print("Gradient of y[0] (2):\n",jacobian(x),"\n")
    jacobian = torch.func.grad(lambda x:func_2(x)[1])
    print("Gradient of y[1] (2):\n",jacobian(x),"\n")
except:
    print("Gradient of y[0] (2):\n",None,"\n")
    print("Gradient of y[1] (2):\n",None,"\n")

try :
    # this should DO work only with 'hstack', not with 'vstack'
    jacobian = torch.func.grad(lambda x:func_3(x)[0])
    print("Gradient of y[0] (2):\n",jacobian(x),"\n")
    jacobian = torch.func.grad(lambda x:func_3(x)[1])
    print("Gradient of y[1] (2):\n",jacobian(x),"\n")
except:
    print("Gradient of y[0] (3):\n",None,"\n")
    print("Gradient of y[1] (3):\n",None,"\n")

# Using torch.autograd.grad
# try :
#     y = outputs=func_1(x)
#     jacobian = torch.autograd.grad(y.sum(),inputs=x)#,grad_outputs=torch.empty_like(y[0]))
#     print("Gradient of y[0] (1):\n",jacobian,"\n")
#     jacobian = torch.autograd.grad(y[1],inputs=x,grad_outputs=torch.empty_like(y[1]))
#     print("Gradient of y[1] (1):\n",jacobian,"\n")
# except:
#     print("Gradient of y[0] (1):\n",None,"\n")
#     print("Gradient of y[1] (1):\n",None,"\n")

# try :
#     x = torch.tensor([5.0, 7.0], requires_grad=True)
#     jacobian = torch.autograd.grad(func_2(x), x)
#     print("Jacobian Matrix (2):\n",jacobian,"\n")
# except:
#     print("Jacobian Matrix (1):\n",None,"\n")

# try :
#     x = torch.tensor([5.0, 7.0], requires_grad=True)
#     jacobian = torch.autograd.grad(func_3(x), x)
#     print("Jacobian Matrix (3):\n",jacobian,"\n")
# except:
#     print("Jacobian Matrix (1):\n",None,"\n")

print()
