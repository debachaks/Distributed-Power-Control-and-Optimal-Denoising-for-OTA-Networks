import numpy as np
import pandas as pd
import numpy.typing as npt

# Using different payoff function

def utility_vector(P: npt.NDArray[np.float64], h: npt.NDArray[np.float64], c: np.float64, eta: np.float64):
    utilities = h * np.sqrt(P) / np.sqrt(eta) - c * np.power(h, 2) * P / eta
    return utilities


def closed_form_p_star(h: npt.NDArray[np.float64], c: np.float64, eta: np.float64, P_max: npt.NDArray[np.float64]):
    base = np.sqrt(eta) / (2 * c * h)
    P_star = np.clip(np.power(base, 2), 1e-10, P_max)  # clipping
    return P_star


def get_best_c(h: npt.NDArray[np.float64], P_max: npt.NDArray[np.float64], eta: np.float64):
    c_vec = np.zeros(len(h))
    for i in range(len(h)):
        c_vec[i] = np.sqrt(eta) / (2 * h[i] * np.sqrt(P_max[i]))
    c_min = np.max(c_vec)
    return c_min


def adaptive_best_response(P: npt.NDArray[np.float64], h: npt.NDArray[np.float64], eta: np.float64, epsilon: np.float64, P_max: npt.NDArray[np.float64], max_iteration: int = 1000):
    c_min = 2 * get_best_c(h, P_max, eta)
    print("Starting c_min:", c_min)

    found = False
    history_util = {}
    history_power = {}

    while not found and c_min < 1e15:

        print(f"\n=== Testing c_min = {c_min:.3e} ===")

        P_current = P.copy()
        utility_history = [utility_vector(P_current, h, c_min, eta)]
        power_history = [P_current.copy()]
        convergence = 0
        iteration = 0
        stop_due_to_negative = False

        while convergence == 0:
            print(iteration)
            iteration += 1
            P_new = closed_form_p_star(h, c_min, eta, P_max)
            utilities_new = utility_vector(P_new, h, c_min, eta)
            utility_history.append(utilities_new.copy())
            power_history.append(P_new.copy())

            if np.any(utilities_new < 0):
                print(f"⚠ Negative utility at iteration {iteration}, increasing c_min...")
                stop_due_to_negative = True
                break

            if np.all(np.abs(P_new - P_current) < epsilon) or iteration == max_iteration:
                convergence = 1
            else:
                P_current = P_new

        history_util[c_min] = np.array(utility_history)
        history_power[c_min] = np.array(power_history)

        if not stop_due_to_negative:
            found = True
            print(f"✅ Found feasible c_min = {c_min}, converged in {iteration} iterations.")
            df = pd.DataFrame({
                "Node": np.arange(1, len(P) + 1),
                "Final Power P*": P_new,
                "Final Utility": utilities_new,
                "Channel Gain h": h
            })
            return c_min, history_util, history_power, df

        c_min *= 2

    print("Could not find a feasible c_min up to 1e15")
    return None, history_util, history_power, None
