import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

#Load JSON Files ----------

def load_results(filepath):
    """Load a single experiment JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


BASE_DIR = "/Users/apple/Desktop/Msc Project/Artefact/project-main/GRAPHS/plot comparison"

FILES = {
    "Honest Baseline":       os.path.join(BASE_DIR, "BASELINE - 20 clients.json"),
    "Label Flipping Attack": os.path.join(BASE_DIR, "ATTACK - 20 Clients.json"),
    "ADL Trust Protocol":    os.path.join(BASE_DIR, "TRUST - 20 clients.json"),
}

COLOURS = {
    "Honest Baseline":       "#b6ff97",
    "Label Flipping Attack": "#e74c3c",
    "ADL Trust Protocol":    "#0097b2",
}

STYLES = {
    "Honest Baseline":       "-",
    "Label Flipping Attack": "--",
    "ADL Trust Protocol":    "-.",
}

#  ----------Load All Data ----------

data = {}
for label, filepath in FILES.items():
    if os.path.exists(filepath):
        data[label] = load_results(filepath)
        print(f"Loaded: {filepath}")
    else:
        print(f"WARNING: {filepath} not found — skipping.")

if not data:
    raise FileNotFoundError("No JSON result files found. Run your experiments first.")

#  ---------- Plot 1: Accuracy Per Round ----------

fig, ax = plt.subplots(figsize=(10, 6))

for label, result in data.items():
    accuracy = result["accuracy_per_round"]
    rounds   = list(range(1, len(accuracy) + 1))

    ax.plot(
        rounds,
        [acc * 100 for acc in accuracy],
        color     = COLOURS[label],
        linestyle = STYLES[label],
        linewidth = 2.2,
        marker    = "o",
        markersize= 4,
        label     = label
    )

for label, result in data.items():
    final_acc = result["accuracy_per_round"][-1] * 100
    ax.axhline(
        y         = final_acc,
        color     = COLOURS[label],
        linestyle = ":",
        linewidth = 0.8,
        alpha     = 0.4
    )

ax.set_title(
    "Federated Learning Accuracy: Baseline vs Attack vs ADL Protocol",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("Communication Round", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_ylim(0, 105)
ax.set_xlim(1, max(len(v["accuracy_per_round"]) for v in data.values()))
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, linestyle="--", alpha=0.4)
ax.tick_params(axis="both", labelsize=10)

for label, result in data.items():
    final_acc   = result["accuracy_per_round"][-1] * 100
    final_round = len(result["accuracy_per_round"])
    ax.annotate(
        f"{final_acc:.2f}%",
        xy         = (final_round, final_acc),
        xytext     = (-40, 8),
        textcoords = "offset points",
        fontsize   = 9,
        color      = COLOURS[label],
        fontweight = "bold"
    )

plt.tight_layout()
plt.savefig("comparison_accuracy.png", dpi=150)
plt.show()
print("Saved: comparison_accuracy.png")

# ── Plot 2: Final Round Bar Chart ─────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(8, 5))

labels      = list(data.keys())
final_accs  = [data[l]["accuracy_per_round"][-1] * 100 for l in labels]
bar_colours = [COLOURS[l] for l in labels]

bars = ax2.bar(labels, final_accs, color=bar_colours, width=0.45, edgecolor="white", linewidth=1.2)

for bar, val in zip(bars, final_accs):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.2f}%",
        ha         = "center",
        va         = "bottom",
        fontsize   = 11,
        fontweight = "bold"
    )

ax2.set_title(
    "Final Round Accuracy — Experimental Comparison",
    fontsize=14, fontweight="bold", pad=15
)
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_ylim(0, 110)
ax2.tick_params(axis="x", labelsize=10)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("comparison_bar.png", dpi=150)
plt.show()
print("Saved: comparison_bar.png")

#  ---------- Plot 3: Accuracy Gap Recovery  ----------

if "Label Flipping Attack" in data and "ADL Trust Protocol" in data:

    attack_acc = data["Label Flipping Attack"]["accuracy_per_round"]
    trust_acc  = data["ADL Trust Protocol"]["accuracy_per_round"]
    min_rounds = min(len(attack_acc), len(trust_acc))

    gap    = [(trust_acc[i] - attack_acc[i]) * 100 for i in range(min_rounds)]
    rounds = list(range(1, min_rounds + 1))

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    ax3.fill_between(rounds, gap, color="#0097b2", alpha=0.25)
    ax3.plot(rounds, gap, color="#0097b2", linewidth=2.2, marker="o", markersize=4)
    ax3.axhline(y=0, color="grey", linewidth=1, linestyle="--")

    ax3.set_title(
        "Accuracy Recovery: ADL Protocol vs Label Flipping Attack (per round)",
        fontsize=13, fontweight="bold", pad=15
    )
    ax3.set_xlabel("Communication Round", fontsize=12)
    ax3.set_ylabel("Accuracy Improvement (%)", fontsize=12)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.savefig("comparison_gap.png", dpi=150)
    plt.show()
    print("Saved: comparison_gap.png")

else:
    print("Gap plot skipped ")

#  ----------Summary Table  ----------

def plot_trust_weights(trust_history, malicious_clients, filename="/Users/apple/Desktop/Msc Project/Artefact/project-main/results/iid - 100 rounds FINAL/results_trust - 20 clients - 100 rounds.json"):
    import matplotlib.pyplot as plt
    import numpy as np

    with open("results_trust_-_20_clients_-_100_rounds.json") as f:
        data = json.load(f)
        
        
    trust_history=data["trust_history"]
    malicious_clients=data["malicious_clients"]

    
    fig, ax = plt.subplots(figsize=(14, 6))

    malicious_set = set(str(c) for c in malicious_clients)
    
    plotted_honest    = False
    plotted_malicious = False

    for client_id, weights in trust_history.items():
        rounds = list(range(1, len(weights) + 1))
        is_malicious = client_id in malicious_set

        if is_malicious:
            label = "Malicious Clients" if not plotted_malicious else "_nolegend_"
            ax.plot(rounds, weights,
                    color="#e74c3c", linewidth=1.5,
                    alpha=0.8, label=label)
            plotted_malicious = True
        else:
            label = "Honest Clients" if not plotted_honest else "_nolegend_"
            ax.plot(rounds, weights,
                    color="#2ecc71", linewidth=1.5,
                    alpha=0.8, label=label)
            plotted_honest = True

    # Mark the minimum weight floor
    ax.axhline(y=0.1, color='gray', linestyle='--',
               linewidth=1.0, alpha=0.6, label='Min Weight Floor (0.1)')
    ax.axhline(y=1.0, color='black', linestyle=':',
               linewidth=1.0, alpha=0.4, label='Neutral Weight (1.0)')

    ax.set_title(
        "ADL Protocol — Client Trust Weight Trajectories (IID, 40% Byzantine)",
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Trust Weight", fontsize=11)
    ax.set_ylim(0, 2.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  [PLOT] saved - {filename}")