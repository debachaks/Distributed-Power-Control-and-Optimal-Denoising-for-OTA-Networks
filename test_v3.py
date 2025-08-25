import numpy as np
import matplotlib.pyplot as plt
import channel_modelling_3GPP as cm
import bestresponsev3_aug25 as br

eta = np.float64(1)
epsilon = np.float64(1e-20)
N = 5
max_iteration = 100
receiver_coordinates = (0, 0, 0)
transmitter_coordinates = [
    (7.5, 3.0, 2.25),
    (12.0, 6.0, 3.0),
    (16.5, 8.25, 3.75),
    (21.0, 10.5, 4.5),
    (25.5, 12.0, 5.25)
]
P_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

P = np.random.uniform(0.1, 1, N)  # random initial power
h = np.array([cm.get_channel_gain(tx, receiver_coordinates, 2405e6) for tx in transmitter_coordinates])

print("Channel gain values are:", h)
print("Initial Power values are:", P)

c_star, utility_histories, power_histories, df = br.adaptive_best_response(
    P, h, eta, epsilon, P_max, max_iteration
)

if c_star is None:
    print("No feasible c_min found")
    exit()

# ----- Plot Utility Behaviour -----
plt.figure(figsize=(10, 6))
for node in range(utility_histories[c_star].shape[1]):
    plt.plot(utility_histories[c_star][:, node], label=f"Node {node}")
plt.xlabel("Iteration")
plt.ylabel("Utility")
plt.title(f"Utility Behaviour for c_min = {c_star}")
plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- Plot Power Behaviour -----
plt.figure(figsize=(10, 6))
for node in range(power_histories[c_star].shape[1]):
    plt.plot(power_histories[c_star][:, node], label=f"Node {node}")
plt.xlabel("Iteration")
plt.ylabel("Power")
plt.title(f"Power Behaviour for c_min = {c_star}")
plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

print(df)
