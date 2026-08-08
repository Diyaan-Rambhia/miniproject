"""
Phase: Adversarial Attack Generation
Expects: PyTorch model instance, clean input tensor x, target labels y, attack strength hyperparams (epsilon, alpha, steps), loss criterion
Outputs: Adversarially perturbed input tensor x_adv
"""

import torch


def fgsm_attack(model, x, y, epsilon, criterion):
    """
    Fast Gradient Sign Method — single-step perturbation.
    """
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    perturbation = epsilon * x.grad.sign()
    x_adv = (x + perturbation).detach()
    return x_adv


def pgd_attack(model, x, y, epsilon, alpha, num_steps, criterion):
    """
    Projected Gradient Descent — multi-step perturbation.
    """
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            perturbation = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = (x_orig + perturbation).detach()

    return x_adv
