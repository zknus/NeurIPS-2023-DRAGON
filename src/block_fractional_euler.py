import torch
import math



import scipy.special as sp
# Helper function to compute binomial coefficients
def binom(alpha, k):
    return sp.gamma(alpha + 1) / (sp.gamma(k + 1) * sp.gamma(alpha - k + 1))



def GL_order_n(alpha, coefficient, f, y0, tspan, device):
    N = len(tspan)
    # h = (tspan[-1] - tspan[0]) / (N - 1)
    h = (tspan[1] - tspan[0])
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device,requires_grad=False)
    # coeff_tensor = torch.tensor(coefficient, dtype=torch.float32, device=device,requires_grad=False)
    coeff_tensor = torch.stack([p for p in coefficient])

    # Precompute coeff_alpha
    coeff_alpha = []
    for a in alpha:
        coeff_alpha.append(torch.tensor([(-1) ** k * binom(a, k) for k in range(N+1)], dtype=torch.float32, device=device))
    coeff_alpha = torch.stack(coeff_alpha)

    y_history = [y0]
    yn = y0

    res = torch.sum( (1/h) ** alpha_tensor)
    res = 1 / res


    # solution = torch.empty(len(tspan), *y0.shape, dtype=y0.dtype, device=y0.device)
    for k in range(1, N):
        tn = tspan[k]
        right = torch.zeros(len(alpha), *y0.shape, dtype=torch.float32, device=device)

        # if k > 0:
        # Vectorized computation of right
        y_history_tensor = torch.stack(y_history)

        # print("y_history_tensor[:k]:", y_history_tensor[:k].shape)
        # print("y_history_tensor[:k].flip(dims=[0]):", y_history_tensor[:k].flip(dims=[0]).shape)
        for i in range(len(alpha)):
            # print("coeff_alpha[i, 1:k+1]:", coeff_alpha[i, 1:k+1].shape)
            # print("coeff_alpha[i, 1:k+1].unsqueeze(1)", coeff_alpha[i, 1:k+1].unsqueeze(1).shape)
            right[i] = (coeff_alpha[i, 1:k+1].view(-1, 1, 1) * y_history_tensor[:k].flip(dims=[0])).sum(dim=0)

        # print("right:", right.shape)
        # print("coeff_tensor.view(-1, 1, 1)", coeff_tensor.view(-1, 1, 1))
        # print("(h ** alpha_tensor).view(-1, 1, 1) ", (h ** alpha_tensor).view(-1, 1, 1) )
        # print("coeff_tensor.view(-1, 1, 1)", coeff_tensor.view(-1, 1, 1).shape)
        right = right / ((h ** alpha_tensor).view(-1, 1, 1) )
        total = (coeff_tensor.view(-1, 1, 1) * right).sum(dim=0)
        # print("total:", total.shape)
        # print("f(tn, yn):", f(tn, yn).shape)



        yn = f(tn, yn)  - total

        yn = yn * res

        # solution[k] = yn
        y_history.append(yn)

    return yn







