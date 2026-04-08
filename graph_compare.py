
    #    ("BASELINE - 20 clients.json",    "#2ecc71", "Honest Baseline"),
    #    ("ATTACK - 20 Clients.json", "#e74c3c", "FedAvg Under Attack"),
     #   ("TRUST - 20 clients.json",      "#3498db", "Trust Protocol"),


import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Load JSON Files ───────────────────────────────────────────────────────────

def load_results(filepath):
    """Load a single experiment JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)

# File paths — adjust if your JSON files live elsewhere
FILES = {
    "Honest Baseline":      "GRAPHS/plot comparison/BASELINE - 20 clients.json",
    "Label Flipping Attack":  "GRAPHS/plot comparison/ATTACK - 20 Clients.json",
    "ADL Trust Protocol":   "GRAPHS/plot comparison/TRUST - 20 clients.json",
}

# Colours for each condition
COLOURS = {
    "Honest Baseline":      "#2ecc71",   # green
    "Label Flipping Attack":  "#e74c3c",   # red
    "ADL Trust Protocol":   "#3498db",   # blue
}

# Line styles
STYLES = {
    "Honest Baseline":      "-",
    "Label Flipping Attack":  "--",
    "ADL Trust Protocol":   "-.",
}

# ── Load All Data ─────────────────────────────────────────────────────────────

data = {}
for label, filepath in FILES.items():
    if os.path.exists(filepath):
        data[label] = load_results(filepath)
        print(f"Loaded: {filepath}")
    else:
        print(f"WARNING: {filepath} not found — skipping.")

if not data:
    raise FileNotFoundError("No JSON result files found. Run your experiments first.")

# ── Plot 1: Accuracy Per Round (Main Comparison) ──────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

for label, result in data.items():
    accuracy = result["accuracy_per_round"]
    rounds   = list(range(1, len(accuracy) + 1))

    ax.plot(
        rounds,
        [acc * 100 for acc in accuracy],   # convert to percentage
        color     = COLOURS[label],
        linestyle = STYLES[label],
        linewidth = 2.2,
        marker    = "o",
        markersize= 4,
        label     = label
    )

# Reference lines for final accuracy
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

# Annotate final accuracy values
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

labels       = list(data.keys())
final_accs   = [data[l]["accuracy_per_round"][-1] * 100 for l in labels]
bar_colours  = [COLOURS[l] for l in labels]

bars = ax2.bar(labels, final_accs, color=bar_colours, width=0.45, edgecolor="white", linewidth=1.2)

# Value labels on top of each bar
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

# ── Plot 3: Accuracy Gap Recovery (CFA vs FedAvg Attack) ─────────────────────

if "FedAvg Under Attack" in data and "CFA Trust Protocol" in data:

    attack_acc = data["FedAvg Under Attack"]["accuracy_per_round"]
    trust_acc  = data["CFA Trust Protocol"]["accuracy_per_round"]
    min_rounds = min(len(attack_acc), len(trust_acc))

    gap = [(trust_acc[i] - attack_acc[i]) * 100 for i in range(min_rounds)]
    rounds = list(range(1, min_rounds + 1))

    fig3, ax3 = plt.subplots(figsize=(10, 5))

    ax3.fill_between(rounds, gap, color="#3498db", alpha=0.25)
    ax3.plot(rounds, gap, color="#3498db", linewidth=2.2, marker="o", markersize=4)
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

# ── Summary Table (printed to console) ───────────────────────────────────────

print("\n" + "=" * 55)
print(f"{'Experiment':<28} {'Final Acc':>10} {'Rounds':>8}")
print("=" * 55)

for label, result in data.items():
    acc    = result["accuracy_per_round"][-1] * 100
    rounds = len(result["accuracy_per_round"])
    print(f"{label:<28} {acc:>9.2f}% {rounds:>8}")

# Gap recovery metric
if "Honest Baseline" in data and "FedAvg Under Attack" in data and "CFA Trust Protocol" in data:
    baseline_acc = data["Honest Baseline"]["accuracy_per_round"][-1] * 100
    attack_acc   = data["FedAvg Under Attack"]["accuracy_per_round"][-1] * 100
    trust_acc    = data["CFA Trust Protocol"]["accuracy_per_round"][-1] * 100

    total_gap    = baseline_acc - attack_acc
    recovered    = trust_acc - attack_acc
    pct_recovery = (recovered / total_gap) * 100 if total_gap > 0 else 0

    print("=" * 55)
    print(f"\nAccuracy gap (Baseline - Attack): {total_gap:.2f}%")
    print(f"ADL recovery:                     {recovered:.2f}%")
    print(f"Gap closed by ADL:                {pct_recovery:.1f}%")
    print("=" * 55)
    
    
if __name__ == "__main__":
    load_results("/Users/apple/Desktop/Msc Project/Artefact/project-main/GRAPH")