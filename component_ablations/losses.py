"""Matched losses: masks never change the remaining branch's denominator."""
import torch


def recovery_terms(predicted, targets, n_real, mask_r, mask_v, coefficient):
    targets = targets.detach()
    errors = (predicted - targets).square()
    zero = predicted.sum() * 0.0
    real = errors[:n_real].mean() if n_real else zero
    virtual = errors[n_real:].mean() if len(targets) > n_real else zero
    total_elements = errors.numel()
    fraction_r = errors[:n_real].numel() / total_elements if total_elements else 0.0
    fraction_v = errors[n_real:].numel() / total_elements if total_elements else 0.0
    lambda_r, lambda_v = coefficient * fraction_r, coefficient * fraction_v
    weighted_r, weighted_v = mask_r * lambda_r * real, mask_v * lambda_v * virtual
    return weighted_r + weighted_v, {
        "recovery_real_raw": real, "recovery_virtual_raw": virtual,
        "recovery_real_weighted": weighted_r, "recovery_virtual_weighted": weighted_v,
        "lambda_R": lambda_r, "lambda_V": lambda_v,
        "recovery_real_tasks": n_real, "recovery_virtual_tasks": len(targets) - n_real,
        "recovery_virtual_skipped": int(len(targets) == n_real),
    }


def critic_terms(real_errors, virtual_errors, quality, weight, mask_p):
    real = real_errors.mean()
    virtual = (quality.detach() * virtual_errors).mean() if virtual_errors.numel() else real * 0.0
    contribution = mask_p * weight * virtual
    return real + contribution, real, virtual, contribution


def actor_terms(real_q, virtual_q, real_bc, virtual_bc, quality, weight,
                mask_p, bc_weight, virtual_bc_enabled):
    alpha = bc_weight / real_q.detach().abs().mean().clamp(min=1e-6)
    real_q_loss = -alpha * real_q.mean()
    virtual_q_loss = -alpha * (quality.detach() * virtual_q).mean() if virtual_q.numel() else real_q_loss * 0.0
    real_bc_loss = real_bc.mean()
    virtual_bc_loss = (quality.detach() * virtual_bc).mean() if virtual_bc.numel() else real_bc_loss * 0.0
    q_loss = real_q_loss + mask_p * weight * virtual_q_loss
    bc_loss = real_bc_loss + mask_p * weight * int(virtual_bc_enabled) * virtual_bc_loss
    return q_loss + bc_loss, {
        "actor_alpha": alpha, "actor_real_q": real_q_loss, "actor_virtual_q_raw": virtual_q_loss,
        "actor_virtual_q_weighted": mask_p * weight * virtual_q_loss,
        "actor_real_bc": real_bc_loss, "actor_virtual_bc_raw": virtual_bc_loss,
        "actor_virtual_bc_weighted": mask_p * weight * int(virtual_bc_enabled) * virtual_bc_loss,
        "policy_loss": q_loss, "bc_loss": bc_loss,
    }
