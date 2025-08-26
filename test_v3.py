import numpy as np
import matplotlib.pyplot as plt
import channel_modelling_3GPP as cm
import bestresponsev5_aug26 as br
import pandas as pd
import math

eta = np.float64(1.0)
epsilon = np.float64(1e-70)
N = 10
max_iteration = 10
receiver_coordinates = (0, 0, 0)

distance = np.arange(1, 100, 10)
transmitter_coordinates = [(np.float64(d)/math.sqrt(3), np.float64(d)/math.sqrt(3), np.float64(d)/math.sqrt(3)) for d in distance]

# transmitter_coordinates = [
#     (2.0, 1.0, 1.5),
#     (3.5, 1.5, 1.8),
#     (5.0, 2.0, 2.0),
#     (6.0, 2.5, 2.1),      
#     (7.5, 3.0, 2.25),
#     (12.0, 6.0, 3.0),
#     (16.5, 8.25, 3.75),
#     (21.0, 10.5, 4.5),
#     (25.5, 12.0, 5.25),
#     (30.0, 15.0, 6.0)
# ]

transmitter_coordinates = sorted(transmitter_coordinates, key=lambda x: math.sqrt(x[0]**2 + x[1]**2 + x[2]**2))

P_max = np.array([np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), 
                   np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0)])

c_init = 1000000000
P = np.float64(np.random.uniform(0.1, 1, N))  # random initial power
h = np.array([cm.get_channel_gain(tx, receiver_coordinates, 2405e6) for tx in transmitter_coordinates])

print("Channel gain values are:", h)
print("Initial Power values are:", P)

c_star, utility_histories, power_histories, df = br.adaptive_best_response(
    P, h, eta, epsilon, P_max, max_iteration, c_init=c_init
)

if c_star is None:
    print("No feasible c_min found")
    exit()

P = power_histories[c_star][-1]
utility = utility_histories[c_star][-1]
distance = np.array([math.sqrt(transmitter_coordinates[i][0]**2 + transmitter_coordinates[i][1]**2 + transmitter_coordinates[i][2]**2) for i in range(N)])

results_df = pd.DataFrame({
    'Node': [i for i in range(N)],
    'Power': P,
    'Utility': utility,
    'Channel Gain': h,
    'Distance': distance
})

pd.set_option('display.float_format', '{:.16f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# ----- Plot Utility Behaviour -----
# plt.figure(figsize=(10, 6))
# for node in range(utility_histories[c_star].shape[1]):
#     plt.plot(utility_histories[c_star][:, node], label=f"Node {node}")
# plt.xlabel("Iteration")
# plt.ylabel("Utility")
# plt.title(f"Utility Behaviour for c_min = {c_star}")
# plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ----- Plot Power Behaviour -----
# plt.figure(figsize=(10, 6))
# for node in range(power_histories[c_star].shape[1]):
#     plt.plot(power_histories[c_star][:, node], label=f"Node {node}")
# plt.xlabel("Iteration")
# plt.ylabel("Power")
# plt.title(f"Power Behaviour for c_min = {c_star}")
# plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.tight_layout()
# plt.show()


mse = np.sum(np.power(h*np.sqrt(P)/np.sqrt(eta)-1, 2))/np.power(N,2)
print(f"MSE: {mse}")
print(h*np.sqrt(P)/np.sqrt(eta)-1)

# for i in range(N):
#     me = h[i]*np.sqrt(P[i])/np.sqrt(eta)
#     other = np.sum(h*np.sqrt(P)/np.sqrt(eta)) - me
#     print(f"NODE {i}: {me}, {other}")
    # print(np.exp(2*other)/(4*c_star))

# print(np.exp(h*np.sqrt(P)/np.sqrt(eta))/(4*c_star))

print("\n" + "="*80)
payoff = np.zeros(N)
for i in range(N):
    me = h[i] * np.sqrt(P[i])
    other = np.sum(h * np.sqrt(P)) -  h[i]*np.sqrt(P[i])
    a = me/np.sqrt(eta) * np.exp(other/np.sqrt(eta)) - c_star * np.power(me/np.sqrt(eta), 2)
    payoff[i] = a
    print(f"NODE {i}: sum of other: {other}, me: {me}, payoff: {a}")

print("\n" + "="*80)
print(br.utility_vector(P, h, c_star, eta))
P[1] = 0.00004
print(br.utility_vector(P, h, c_star, eta))


for i in range(N):
    us = P[i]
    them = eta/h[i]**2
    print(f"NODE {i}: us: {us}, them: {them}")







