import numpy as np
import matplotlib.pyplot as plt

K, T, RUNS = 10, 2000, 500
STRATEGIES = ["stochastic", "systematic", "systematic_random_phase"]


def simulate(strategy, eps, sigma, seed=0, T=T, runs=RUNS, k=K):
    means = np.random.default_rng(seed).normal(5, 1, size=(runs, k))
    best_arm, best_val = means.argmax(1), means.max(1)

    rng = np.random.default_rng(seed + 1)
    Q, N = np.zeros((runs, k)), np.zeros((runs, k))
    S = int(round(1 / eps))

    phase = (rng.integers(0, S, size=runs) if strategy == "systematic_random_phase"
             else np.zeros(runs, dtype=int))

    idx = np.arange(runs)
    optimal = np.zeros((runs, T))
    regret = np.zeros((runs, T))
    avg_reward = np.zeros((runs, T))

    for t in range(T):
        if strategy == "stochastic":
            explore = rng.random(runs) < eps
        else:
            explore = (t + phase) % S == 0

        action = np.where(explore, rng.integers(0, k, size=runs), Q.argmax(1))
        reward = rng.normal(means[idx, action], sigma)

        N[idx, action] += 1
        Q[idx, action] += (reward - Q[idx, action]) / N[idx, action]

        optimal[:, t] = action == best_arm
        regret[:, t] = best_val - means[idx, action]
        avg_reward[:, t] = reward

    return avg_reward, optimal * 100, np.cumsum(regret, axis=1)


def summarize(sigma, eps, seed=0):
    res = {s: simulate(s, eps, sigma, seed) for s in STRATEGIES}
    base_final = res["stochastic"][2][:, -1]
    rows = []
    for s in STRATEGIES:
        _, opt, reg = res[s]
        final = reg[:, -1]
        diff = final - base_final
        rows.append((s, final.mean(), final.std(ddof=1) / np.sqrt(RUNS),
                     diff.mean(), diff.std(ddof=1) / np.sqrt(RUNS),
                     opt[:, -500:].mean()))
    return res, rows


if __name__ == "__main__":
    for sigma in (0.1, 1.0, 2.0):
        for eps in (0.1, 0.01):
            res, rows = summarize(sigma, eps)
            print(f"\nsigma={sigma}  eps={eps}")
            print(f"{'strategy':26s} {'final regret':>16s} {'diff vs stoch (paired)':>28s} {'%opt last500':>13s}")
            for s, m, se, d, dse, o in rows:
                print(
                    f"{s:26s} {m:8.1f} ± {se:5.1f} {d:+12.1f} ± {dse:5.1f} (z={d / dse if dse else 0:+.1f}) {o:10.1f}")


            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            colors = {"stochastic": "red", "systematic": "blue", "systematic_random_phase": "green"}
            x = np.arange(T)

            titles = ["Average Reward", "% Optimal Action", "Cumulative Regret"]

            for i, data_idx in enumerate([0, 1, 2]):  # 0=reward, 1=optimal, 2=regret
                for s in STRATEGIES:
                    data = res[s][data_idx]
                    m, se = data.mean(0), data.std(0, ddof=1) / np.sqrt(RUNS)
                    ax[i].plot(x, m, color=colors[s], label=s, lw=1.5)
                    ax[i].fill_between(x, m - se, m + se, color=colors[s], alpha=0.25)

                ax[i].set_title(f"{titles[i]} (noise={sigma}, eps={eps})")
                ax[i].set_xlabel("Steps")
                ax[i].grid(alpha=0.3)
                ax[i].legend()

            plt.tight_layout()
            plt.savefig(f"final_sigma{sigma}_eps{eps}.png", dpi=150)
            plt.show()
            plt.close()