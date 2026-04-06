#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:25:42 2026

@author: apple
"""

# trust_strategy.py
import numpy as np
import flwr as fl
from flwr.common import (
    Parameters, FitRes, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union

class ADLStrategy(fl.server.strategy.FedAvg):
    """
    Convergence-First Aggregation (CFA) — A selfish trust protocol.

    Core philosophy: the global model's convergence trajectory is the
    sole ground truth for evaluating client update quality.

    Unlike FLTrust or TrustFedAvg, this protocol:
    - Makes NO assumption about client intent
    - Does NOT require a trusted root dataset
    - Does NOT compute per-round cosine similarity against a reference
    - ONLY cares whether the model improved after last round

    If accuracy improved  → reward clients whose updates aligned
    If accuracy degraded  → penalise clients whose updates diverged
    """

    def __init__(
        self,
        num_clients:     int   = 10,
        num_rounds:      int   = 30,
        penalty_factor:  float = 0.85,   # trust multiplier on blame
        reward_factor:   float = 1.10,   # trust multiplier on reward
        min_weight:      float = 0.1,    # floor — never silence a client
        max_weight:      float = 2.0,    # ceiling — prevent dominance
        **kwargs):
        
        
        super().__init__(**kwargs)
        self.num_clients    = num_clients
        self.num_rounds     = num_rounds
        self.penalty_factor = penalty_factor
        self.reward_factor  = reward_factor
        self.min_weight     = min_weight
        self.max_weight     = max_weight

        # Trust weights — start neutral at 1.0
        self.trust_weights: Dict[str, float] = {
            str(i): 1.0 for i in range(num_clients)
        }

        # History for Layer 5 visualisation
        self.trust_history:  Dict[str, List[float]] = {
            str(i): [1.0] for i in range(num_clients)
        }
        self.accuracy_history: List[float] = []
        self.exclusion_log:    List[Dict]  = []

        # State carried across rounds
        self.prev_accuracy:      Optional[float]            = None
        self.prev_client_deltas: Optional[Dict[str, np.ndarray]] = None
        self.prev_global_delta:  Optional[np.ndarray]       = None
        self.current_round:      int                        = 0

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _flatten(self, params: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([p.flatten() for p in params])

    def _cosine_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))

    # ── Retrospective Blame / Reward ──────────────────────────────────────────

    def _retrospective_update(
        self,
        current_accuracy: float,
        client_ids: List[str],) -> Dict[str, str]:
        """
        Core CFA logic — called at the START of each round (t)
        using accuracy from round (t-1).

        Returns a dict of client_id -> verdict for logging.
        """
        verdicts = {}

        if self.prev_accuracy is None or self.prev_client_deltas is None:
            # Round 1 — no history, skip blame/reward
            for cid in client_ids:
                verdicts[cid] = "SKIP (round 1)"
            return verdicts

        delta_acc = current_accuracy - self.prev_accuracy

        if delta_acc >= 0:
            # ── Model improved — reward aligned clients ────────────────────
            print(f"  [CFA] Δacc = +{delta_acc:.4f} → REWARDING aligned clients")
            for cid in client_ids:
                if cid not in self.prev_client_deltas:
                    verdicts[cid] = "NO HISTORY"
                    continue

                # How aligned was this client with the global update direction?
                sim = self._cosine_similarity(
                    self.prev_client_deltas[cid],
                    self.prev_global_delta
                )

                if sim > 0.01:
                    old = self.trust_weights[cid]
                    # Scale reward by alignment strength
                    self.trust_weights[cid] = min(
                        old * (1 + (self.reward_factor - 1) * sim),
                        self.max_weight
                    )
                    verdicts[cid] = f"REWARDED (sim={sim:+.3f})"
                    
                    
                  # mild penalty — model improved overall  
                  
                elif sim < -0.01: 
                    old = self.trust_weights[cid]
                    self.trust_weights[cid] = max(old * 0.98,self.min_weight)
                    verdicts[cid] = f"MILD PENALTY (sim={sim:+.3f})"
                                        
        
                else:
                    verdicts[cid] = f"NEUTRAL  (sim={sim:+.3f})"

        else:
            # ── Model degraded — penalise divergent clients ────────────────
            print(f"  [CFA] Δacc = {delta_acc:.4f} → PENALISING divergent clients")
            
            # Score each client by deviation from global update
            deviation_scores: Dict[str, float] = {}
            for cid in client_ids:
                if cid not in self.prev_client_deltas:
                    deviation_scores[cid] = 0.0
                    continue
                sim = self._cosine_similarity(
                    self.prev_client_deltas[cid],
                    self.prev_global_delta
                )
                # Low or negative similarity = high deviation
                deviation_scores[cid] = 1 - sim

            # Penalise clients above average deviation
            avg_deviation = np.mean(list(deviation_scores.values()))

            for cid in client_ids:
                score = deviation_scores.get(cid, 0.0)
                if score > avg_deviation:
                    old = self.trust_weights[cid]
                    # Scale penalty by how much above average the deviation is
                    excess = score - avg_deviation
                    self.trust_weights[cid] = max(
                        old * (self.penalty_factor - 0.1 * excess),
                        self.min_weight
                    )
                    verdicts[cid] = f"PENALISED (dev={score:.3f})"
                else:
                    verdicts[cid] = f"CLEARED   (dev={score:.3f})"

        return verdicts

    # ── Core Aggregation ──────────────────────────────────────────────────────
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[BaseException, Tuple[ClientProxy, fl.common.EvaluateRes]]],) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Override evaluate aggregation to capture accuracy in real time.
        This feeds accuracy into the CFA blame/reward logic DURING simulation.
        """
        if not results:
            return None, {}
    
        # Weighted average accuracy across all clients
        total_examples = sum(r.num_examples for _, r in results)
        
        weighted_acc   = sum(
            r.num_examples * r.metrics["accuracy"]
            for _, r in results) / total_examples
    
        # Store for next round's blame/reward decision
        self.prev_accuracy = (
            self.accuracy_history[-1]
            if self.accuracy_history else None)
        
        self.accuracy_history.append(weighted_acc)
    
        print(f"  [CFA] Round {server_round} accuracy recorded: {weighted_acc:.4f}"
              f" | Δacc = {weighted_acc - self.prev_accuracy:.4f}"
              if self.prev_accuracy is not None
              else f"  [CFA] Round {server_round} accuracy recorded: {weighted_acc:.4f} (baseline)")
    
        return weighted_acc, {"accuracy": weighted_acc}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[BaseException, Tuple[ClientProxy, FitRes]]],) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        self.current_round = server_round

        if not results:
            return None, {}

        # ── Step 1: Extract client updates ────────────────────────────────────
        client_data = []
        for i, (client_proxy, fit_res) in enumerate(results):
            params    = parameters_to_ndarrays(fit_res.parameters)
            flat      = self._flatten(params)
            client_id = str(fit_res.metrics.get("client_id", f"unknown_{i}"))
            client_data.append({
                "id":     client_id,
                "flat":   flat,
                "params": params,
                "n":      fit_res.num_examples,
            })

        client_ids = [cd["id"] for cd in client_data]

        # ── Step 2: Retrospective update using LAST round's accuracy ──────────
        # Current accuracy isn't known yet — we use what was recorded
        # at the end of the previous round's evaluation
        current_acc = (
            self.accuracy_history[-1]
            if self.accuracy_history else None
        )

        if current_acc is not None:
            verdicts = self._retrospective_update(current_acc, client_ids)
        else:
            verdicts = {cid: "SKIP (no accuracy yet)" for cid in client_ids}

        # ── Step 3: Compute global update direction ───────────────────────────
        # Weighted mean of all updates — used as reference for deviation scoring
        all_flat      = np.array([cd["flat"] for cd in client_data])
        global_delta  = np.mean(all_flat, axis=0)

        # Store per-client deviations from global for NEXT round's blame
        self.prev_client_deltas = {
            cd["id"]: cd["flat"] - global_delta
            for cd in client_data
        }
        self.prev_global_delta = global_delta

        # ── Step 4: Trust-weighted aggregation ───────────────────────────────
        aggregated  = None
        total_weight = 0.0

        print(f"\n[ROUND {server_round}] CFA Trust Weights:")
        for cd in client_data:
            cid    = cd["id"]
            weight = self.trust_weights.get(cid, 1.0)
            verdict = verdicts.get(cid, "")

            print(f"  Client {cid:>2}: weight={weight:.3f}  {verdict}")

            # Log trust history
            self.trust_history[cid].append(round(weight, 4))

            total_weight += weight
            if aggregated is None:
                aggregated = [weight * layer for layer in cd["params"]]
            else:
                for j, layer in enumerate(cd["params"]):
                    aggregated[j] += weight * layer

        # Normalise
        aggregated = [w / total_weight for w in aggregated]

        print(f"  → Total weight: {total_weight:.3f} "
              f"| Avg weight: {total_weight/len(client_data):.3f}")

        return ndarrays_to_parameters(aggregated), {}

    def update_accuracy(self, accuracy: float):
        """
        Called from server after each evaluation round.
        Stores accuracy so next round's aggregate_fit can use it.
        """
        self.prev_accuracy = (
            self.accuracy_history[-1]
            if self.accuracy_history else None
        )
        self.accuracy_history.append(accuracy)

    def get_trust_summary(self) -> Dict:
        return {
            "trust_weights_final": self.trust_weights,
            "trust_history":       self.trust_history,
            "accuracy_history":    self.accuracy_history,
            "exclusion_log":       self.exclusion_log,
        }