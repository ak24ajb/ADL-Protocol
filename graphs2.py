#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:12:45 2026

@author: apple
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 


#this is experimental to check code working
def plot_single_experiment(results, filename="/Users/apple/Desktop/Msc Project/Artefact/project-main/results/attack1"):
 
    label    = results["experiment"]
    accuracy = results["accuracy_per_round"]
    rounds   = list(range(1, len(accuracy) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Colour coding by experiment type
    colours = {
        "honest_baseline":    "#2ecc71",   # green
        "fedavg_under_attack": "#e74c3c",  # red
        "trust_protocol":     "#3498db",   # blue
    }
    colour = colours.get(label, "#95a5a6")

    ax.plot(rounds, accuracy, 
            marker='o', linewidth=2, 
            markersize=4, color=colour, 
            label=label.replace("_", " ").title())

    # Mark malicious clients on the chart if attack experiment
    if results.get("malicious_clients"):
        ax.set_title(
            f"{label.replace('_', ' ').title()}\n"
            f"Malicious Clients: {results['malicious_clients']} "
            f"({len(results['malicious_clients'])/10*100:.0f}% of federation)",
            fontsize=12
        )
    else:
        ax.set_title(label.replace("_", " ").title(), fontsize=12)

    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, len(accuracy) + 1, 10))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if filename is None:
        filename = f"plot_{label}.png"

    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  [PLOT] Saved → {filename}")



#compares all 3 JSON files
#only work in the final stage
def plot_comparison(filename="/Users/apple/Desktop/Msc Project/Artefact/project-main/results/plot_comparison.png"):
  
    experiments = [
        ("results_honest_baseline.json",    "#2ecc71", "Honest Baseline"),
        ("results_fedavg_under_attack.json", "#e74c3c", "FedAvg Under Attack"),
        ("results_trust_protocol.json",      "#3498db", "Trust Protocol"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    found_any = False

    for filepath, colour, label in experiments:
        try:
            with open(filepath) as f:
                data = json.load(f)
            accuracy = data["accuracy_per_round"]
            rounds   = list(range(1, len(accuracy) + 1))
            ax.plot(rounds, accuracy,
                    marker='o', linewidth=2,
                    markersize=4, color=colour,
                    label=label)
            found_any = True
        except FileNotFoundError:
            print(f"  [PLOT] Skipping {filepath} — not found yet")

    if not found_any:
        print("  [PLOT] No result files found. Run experiments first.")
        return

    ax.set_title(
        "Federated Learning — Baseline vs Attack vs Trust Protocol",
        fontsize=13
    )
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  [PLOT] Comparison saved → {filename}")