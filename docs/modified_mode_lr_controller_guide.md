# Modified Mode: Combined LR Controller Guide

This note explains a practical way to update each learning-rate group separately using both:

1. Loss-component signals
2. Per-group gradient signals

The goal is to keep training stable while making updates more targeted than a single global-loss rule.

## Runtime scope map (current implementation)

- `core_individual`: updates LR groups only; uses `tune_min_improvement` gate.
- `core_only`: updates LR groups plus only `grow_grad2d`; no adaptive gate requirement.
- `core_ai_optimization`: updates LR groups plus only `grow_grad2d`; uses `tune_min_improvement` gate.
- `core_individual_plus_strategy`: updates LR groups plus full strategy (`grow_grad2d`, `prune_opa`, `refine_every`, `reset_every`).
- Legacy alias: `with_strategy` is normalized to `core_individual_plus_strategy`.

## 1) Total loss and component losses

Training usually has multiple loss terms:

- $L_1$: pixel reconstruction loss
- $L_{ssim}$: structural similarity loss
- $L_{depth}$: depth loss (optional)
- $L_{opa\_reg}$: opacity regularization
- $L_{scale\_reg}$: scale regularization

Total loss:

$$
L_{tot} = L_1 + \lambda_{ssim}L_{ssim} + \lambda_{depth}L_{depth} + \lambda_{opa}L_{opa\_reg} + \lambda_{scale}L_{scale\_reg}
$$

Where each $\lambda$ controls how strongly that component contributes.

## 2) Normalize each loss component

Raw components are on different scales, so normalize each against its EMA baseline:

$$
z_k(t) = \log\left(\frac{L_k(t)+\epsilon}{EMA[L_k](t)+\epsilon}\right)
$$

Meaning:

- $z_k > 0$: component is above recent baseline (more pressure)
- $z_k < 0$: component is below baseline (less pressure)

## 3) Per-group gradient signal

For each parameter group $g \in \{means, opacities, scales, quats, sh0, shN\}$:

$$
G_g(t) = \left\|\nabla_{\theta_g}L_{tot}(t)\right\|_2
$$

Normalize against EMA:

$$
\hat g_g(t) = \log\left(\frac{G_g(t)+\epsilon}{EMA[G_g](t)+\epsilon}\right)
$$

Interpretation:

- $\hat g_g > 0$: group is under stronger-than-usual optimization pressure
- $\hat g_g < 0$: weaker-than-usual pressure

## 4) Build one pressure score per LR group

Combine normalized component signals and gradient signal:

$$
P_g(t) = \sum_k w_{g,k}z_k(t) + \alpha_g\hat g_g(t)
$$

Where:

- $w_{g,k}$: influence of loss component $k$ on group $g$
- $\alpha_g$: weight of gradient term for group $g$

## 5) Convert pressure into LR multiplier

$$
m_g(t) = clip\left(\exp(-\eta_g P_g(t)),\ m_g^{min},\ m_g^{max}\right)
$$

- Positive pressure usually lowers LR ($m_g < 1$)
- Negative pressure can raise LR ($m_g > 1$)

## 6) Smooth and clamp LR update

Smooth update:

$$
lr_g^{new} = (1-\rho)lr_g^{old} + \rho\left(lr_g^{old}\cdot m_g(t)\right)
$$

Hard clamp:

$$
lr_g^{new} = clip\left(lr_g^{new},\ lr_{g,min},\ lr_{g,max}\right)
$$

## 7) Suggested initial mapping weights

Example starting weights $w_{g,k}$:

### Opacity group

- $w_{opa,L1}=0.30$
- $w_{opa,Lssim}=0.20$
- $w_{opa,Ldepth}=0.10$
- $w_{opa,Lopa\_reg}=0.40$
- $w_{opa,Lscale\_reg}=0.00$

### Means group

- $w_{means,L1}=0.45$
- $w_{means,Lssim}=0.25$
- $w_{means,Ldepth}=0.25$
- $w_{means,Lopa\_reg}=0.00$
- $w_{means,Lscale\_reg}=0.05$

### Scales group

- $w_{scales,L1}=0.25$
- $w_{scales,Lssim}=0.15$
- $w_{scales,Ldepth}=0.10$
- $w_{scales,Lopa\_reg}=0.00$
- $w_{scales,Lscale\_reg}=0.50$

### Quats group

- $w_{quats,L1}=0.35$
- $w_{quats,Lssim}=0.25$
- $w_{quats,Ldepth}=0.30$
- $w_{quats,Lopa\_reg}=0.00$
- $w_{quats,Lscale\_reg}=0.10$

### SH groups (sh0/shN)

- $w_{sh,L1}=0.55$
- $w_{sh,Lssim}=0.40$
- $w_{sh,Ldepth}=0.05$
- $w_{sh,Lopa\_reg}=0.00$
- $w_{sh,Lscale\_reg}=0.00$

## 8) Stable defaults

- Controller gain: $\eta_g = 0.15$
- Gradient blend: $\alpha_g = 0.5$
- Smoothing: $\rho = 0.25$
- Per-step multiplier bounds: $m_g^{min}=0.9$, $m_g^{max}=1.1$
- Update interval: every 25 steps
- Warmup: no adaptive updates for first 200 steps
- Deadband: if $|P_g| < 0.03$, keep LR unchanged

## 9) Why this is better than one global-loss rule

A single global loss often cannot tell which parameter group needs change.

This combined approach gives each group its own update signal:

1. Component-level pressure (what kind of error is rising)
2. Group-level gradient pressure (which group is currently being pushed)

That usually gives better control and less overcorrection.

## 10) Practical note for Google Docs

Google Docs does not render LaTeX math in normal text by default.

To keep formulas preserved:

1. Paste this markdown as plain text
2. Insert equations in Docs using `Insert -> Equation`
3. Paste each formula into equation boxes (or use a Docs add-on that supports LaTeX)
