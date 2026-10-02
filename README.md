# Systematic ε-Greedy: A Deterministic Periodic Exploration Schedule

An empirical comparison of deterministic vs. stochastic exploration in stationary Multi-Armed Bandits. 

## 📄 Read the Paper
[Download the full technical report (PDF)](./paper.pdf)

## 🧠 Abstract
The standard ε-greedy implementation explores randomly, making the timing and total count of exploratory pulls variable. This project proposes **Systematic ε-greedy**, a deterministic alternative that explores on a fixed clock (every $S = \lfloor 1/\epsilon \rfloor$ steps), alongside a phase-randomized extension. 

Evaluated on a stationary 10-armed Gaussian testbed ($\sigma \in \{0.1, 1, 2\}$, $\epsilon \in \{0.1, 0.01\}$, $T=2,000$) across 4,000 paired environments, the phase-randomized version yields a modest, statistically clear regret reduction (6–10% at $\epsilon = 0.01$). The project also identifies and resolves a measurement artifact where synchronized exploration creates false visual superiority in raw per-step curves.

## 🚀 How to Run
```bash
pip install numpy matplotlib
python bandit_final.py
