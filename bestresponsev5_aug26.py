import numpy as np
import pandas as pd
import numpy.typing as npt
from scipy.optimize import minimize

# Asynchronous Best Response Implementation
# Using different payoff function

def utility_vector(P: npt.NDArray[np.float64], h: npt.NDArray[np.float64], c: np.float64, eta: np.float64):
    utilities = np.zeros(len(P))
    sum1 = np.sum(h * np.sqrt(P) / np.sqrt(eta))

    for i in range(len(P)):
        me = h[i] * np.sqrt(P[i]) / np.sqrt(eta)
        other = sum1 - me
        utilities[i] = me * np.exp(other) - c * np.power(me, 2)
    return utilities


def closed_form_p_star(h: npt.NDArray[np.float64], c: np.float64, eta: np.float64, P_max: npt.NDArray[np.float64]):
    # sum1 = np.sum(h * np.sqrt(P_max)/np.sqrt(eta))
    # P_star = np.zeros(len(P_max))
    # for i in range(len(P_max)):
    #     me = h[i] * np.sqrt(P_star[i]) / np.sqrt(eta)
    #     other = sum1 - me
    #     P_star[i] = eta * np.exp(2*other)/ (4 * np.power(c, 2) * np.power(h[i], 2))

    P_init = P_max.copy()
    P_star = minimize(lambda x: -utility_vector(x, h, c, eta), P_init, method='BFGS')
    P_star = np.clip(P_star, 0, P_max)
    return P_star


def get_best_c(h: npt.NDArray[np.float64], P_max: npt.NDArray[np.float64], eta: np.float64):
    pass


def adaptive_best_response(P: npt.NDArray[np.float64], h: npt.NDArray[np.float64], eta: np.float64, epsilon: np.float64, P_max: npt.NDArray[np.float64], max_iteration: int = 1000, c_init: np.float64 = 10000000):
    c = np.float64(c_init)
    print("Starting c:", c)

    found = False
    history_util = {}
    history_power = {}

    while not found and c <= 1e60 and c >= 1e-60:

        print(f"\n=== Testing c = {c} ===")

        P_current = P.copy()
        utility_history = [utility_vector(P_current, h, c, eta)]
        power_history = [P_current.copy()]
        convergence = 0
        iteration = 0
        stop_due_to_negative = False
        stop_due_to_max = False

        while convergence == 0:
            iteration += 1
        
            P_new = P_current.copy()
            for i in range(len(P_new)):
                sum1 = np.sum(h * np.sqrt(P_new)/np.sqrt(eta))
                me = h[i] * np.sqrt(P_new[i]) / np.sqrt(eta)
                other = sum1 - me
                P_new[i] = eta * np.exp(2*other)/ (4 * np.power(c, 2) * np.power(h[i], 2))
            
            P_new = np.clip(P_new, 0, P_max)
            utility_history.append(utility_vector(P_new, h, c, eta))
            power_history.append(P_new.copy())

            if np.any(utility_history[-1] < 0):
                print(f"⚠ Negative utility at iteration {iteration}, increasing c...")
                stop_due_to_negative = True
                break

            if np.all(P_new == P_max):
                print(f"⚠ All powers are at maximum at iteration {iteration}, increasing c...")
                stop_due_to_max = True
                break
            
            if iteration == max_iteration or np.all(np.abs(utility_history[-1] - utility_history[-2]) < epsilon):
                convergence = 1
            else:
                P_current = P_new

        history_util[c] = np.array(utility_history)
        history_power[c] = np.array(power_history)

        if not stop_due_to_negative and not stop_due_to_max:
            found = True
            print(f"✅ Found feasible c = {c}, converged in {iteration} iterations.")
            df = pd.DataFrame({
                "Node": np.arange(1, len(P) + 1),
                "Final Power P*": P_new,
                "Final Utility": utility_history[-1],
                "Channel Gain h": h
            })
            return c, history_util, history_power, df

        if stop_due_to_negative:
            c/=2
            break
        c*=2

    print("Could not find a feasible c")
    return None, history_util, history_power, None
