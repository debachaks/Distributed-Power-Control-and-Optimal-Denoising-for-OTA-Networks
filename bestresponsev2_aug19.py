import numpy as np
import pandas as pd
import numpy.typing as npt


def utility_vector(P, h, c_max, eta):
    utilities = np.zeros(len(P))
    P = np.asarray(P)
    for n in range(len(P)):
        gain_term = (h[n] * np.sqrt(P[n]) / np.sqrt(eta)) * \
                    np.sum([np.sqrt(P[m]) * h[m] / np.sqrt(eta) for m in range(len(P)) if m != n])
        penalty_term = (c_max * h[n]**2 * P[n] / eta) * \
                       np.sum([P[m] * h[m]**2 / eta for m in range(len(P))])
        utilities[n] = gain_term - penalty_term
    return utilities


def closed_form_p_star(P, h, c_max, eta, P_max_vec):
    optimal_power = []
    sqrtP = np.sqrt(P)
    h2 = h ** 2
    for i in range(len(P)):
        mult = eta / (2 * c_max * h[i])
        sum1 = np.sum(h[j] * sqrtP[j] for j in range(len(P)) if i != j)
        sum2 = np.sum(h2[j] * P[j] for j in range(len(P)) if i != j) + 1e-12
        base = mult * (sum1 / sum2)
        P_star = np.clip(base ** 2, 1e-10, P_max_vec[i])  # clipping
        optimal_power.append(P_star)
    return np.array(optimal_power)


def adaptive_best_response(P: npt.NDArray[np.float64], h: npt.NDArray[np.float64], eta: float, epsilon: float, P_max: npt.NDArray[np.float64], max_iteration: int = 1000):
    h1, h2 = np.sort(h)[-2:]
    c_max = 6.586e+11 #eta / (4 * h1 * h2)
    print("Starting c_max:", c_max)

    P_max_vec = np.ones(len(P)) * P_max
    factor = 2.0
    found = False
    history_util = {}
    history_power = {}

    while not found and c_max < 1e15:

        print(f"\n=== Testing c_max = {c_max:.3e} ===")

        P_current = P.copy()
        utility_history = [utility_vector(P_current, h, c_max, eta)]
        power_history = [P_current.copy()]
        convergence = 0
        iteration = 0
        stop_due_to_negative = False

        while convergence == 0:
            print(iteration)
            iteration += 1
            P_new = closed_form_p_star(P_current, h, c_max, eta, P_max_vec)
            utilities_new = utility_vector(P_new, h, c_max, eta)
            utility_history.append(utilities_new.copy())
            power_history.append(P_new.copy())

            if np.any(utilities_new < 0):
                print(f"⚠ Negative utility at iteration {iteration}, increasing c_max...")
                stop_due_to_negative = True
                break

            if np.linalg.norm(P_new - P_current) < epsilon or iteration == max_iteration:
                convergence = 1
            else:
                P_current = P_new

        history_util[c_max] = np.array(utility_history)
        history_power[c_max] = np.array(power_history)

        if not stop_due_to_negative:
            found = True
            print(f"✅ Found feasible c_max = {c_max:.3e}, converged in {iteration} iterations.")
            df = pd.DataFrame({
                "Node": np.arange(1, len(P) + 1),
                "Final Power P*": P_new,
                "Final Utility": utilities_new,
                "Channel Gain h": h
            })
            print(df)
            return c_max, history_util, history_power, df

        c_max *= factor

    print("Could not find a feasible c_max up to 1e15")
    return None, history_util, history_power, None
