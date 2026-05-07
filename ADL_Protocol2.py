#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:25:42 2026

@author: apple
"""

import numpy as np
import flwr as fl
from flwr.common import (Parameters, FitRes, Scalar, ndarrays_to_parameters, parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union

class ADLStrategy(fl.server.strategy.FedAvg):

    def __init__(self, num_clients:int=20, num_rounds:int=10,
                 
        penalty_factor:  float = 0.85,   #trust multiplier on blame
        reward_factor:   float = 1.10,   # ^ for reward
        #scoring bounds
        min_weight:      float = 0.1,   
        max_weight:      float = 2.0, **kwargs):
        
#fedavg inst-------------
        super().__init__(**kwargs)
        self.num_clients    = num_clients
        self.num_rounds     = num_rounds
        self.penalty_factor = penalty_factor
        self.reward_factor  = reward_factor
        self.min_weight     = min_weight
        self.max_weight     = max_weight

        #starting neutral here
        self.trust_weights: Dict[str, float] = {str(i): 1.0 for i in range(num_clients)}

        #saving history here
        self.trust_history:  Dict[str, List[float]] = {str(i): [1.0] for i in range(num_clients)}
        self.accuracy_history: List[float] = []
        self.exclusion_log:    List[Dict]  = []

        #state carried across rounds, added
        self.prev_accuracy:      Optional[float] = None
        self.prev_client_deltas: Optional[Dict[str, np.ndarray]] = None
        self.prev_global_delta:  Optional[np.ndarray] = None
        self.current_round:      int = 0

 #-------------loaded utilities here-------------
 
 #Flattening of the 3-layer MLP arrays into 1 long vector for COS SIM

    def _flatten(self, params):
        return np.concatenate([p.flatten() for p in params])
    

    def _cosine_similarity(self, u: np.ndarray, v: np.ndarray):
        
        #computing the magnitude of the vector through linear algebra
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        
        #fixed after error 1
        if nu == 0 or nv == 0:
            return 0.0
        
        #computing and clipping the results in the range
        return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))



#-----------------------

#   def _compute_reference(self, updates: List[np.ndarray]) -> np.ndarray:

#        return np.mean(updates, axis=0)

 #   def _ema_update(self, client_id: str, cos_sim: float) -> float:
  
#        normalised        = (cos_sim + 1) / 2
        
  #       If client_id not recognised, initialise it at neutral trust
#        if client_id not in self.trust_scores:
 #           self.trust_scores[client_id]  = 0.5
 #           self.trust_history[client_id] = []
            
            
 #       old               = self.trust_scores[client_id]
  #      updated           = self.alpha * normalised + (1 - self.alpha) * old
  #      self.trust_scores[client_id] = updated
   #     self.trust_history[client_id].append(round(updated, 4))
  #      return updated

#    def _late_round_penalty(self, trust: float) -> float:

#        progress = self.current_round / self.num_rounds
#        penalty  = 1 - (progress * self.beta * (1 - trust))
#        return trust * max(penalty, 0.0)
    


    #-------------Blame , Reward core -------------

    def _retrospective_update(self, current_accuracy: float, client_ids: List[str]):
      
        #------- Core Logic implemented here,-------------
        verdicts = {}

        if self.prev_accuracy is None or self.prev_client_deltas is None:
            #skip for round 1
            for cid in client_ids:
                verdicts[cid] = "SKIP (round 1)"
            return verdicts

        delta_acc = current_accuracy - self.prev_accuracy



#Model improved — reward aligned clients -------------
        if delta_acc >= 0:
            
            print(f"  acc = +{delta_acc:.4f} - REWARDING good clients")
            for cid in client_ids:
                if cid not in self.prev_client_deltas:
                    verdicts[cid] = "NO HISTORY YET"
                    continue

                # aligned with global update direction?
                sim = self._cosine_similarity(self.prev_client_deltas[cid], self.prev_global_delta)

                if sim > 0.01:
                    old = self.trust_weights[cid]
                    # Scale reward by alignment strength
                    self.trust_weights[cid] = min(old * (1 + (self.reward_factor - 1) * sim), self.max_weight)
                    verdicts[cid] = f"REWARDED (sim={sim:+.3f})"
                    
                  #introducing here the mild penalty even when model improved overall  
                  #To ensure fairness regardless of the results
                  
                elif sim < -0.01: 
                    old = self.trust_weights[cid]
                    self.trust_weights[cid] = max(old * 0.98,self.min_weight)
                    verdicts[cid] = f"MILD PENALTY (sim={sim:+.3f})"
                                        
                else:
                    verdicts[cid] = f"NEUTRAL  (sim={sim:+.3f})"
                    

#Model degraded — penalise divergent clients -------------
        else:
            print(f"  [ acc = {delta_acc:.4f} - PENALISING bad clients")
            
            # Score each client by deviation 
            deviation_scores: Dict[str, float] = {}
            
            for cid in client_ids:
                if cid not in self.prev_client_deltas:
                    deviation_scores[cid] = 0.0
                    continue
                
                #same as above
                sim = self._cosine_similarity(self.prev_client_deltas[cid], self.prev_global_delta)
                
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
                    self.trust_weights[cid] = max(old * (self.penalty_factor - 0.1 * excess), self.min_weight)
                    
                    verdicts[cid] = f"PENALISED (dev={score:.3f})"
                else:
                    verdicts[cid] = f"CLEARED   (dev={score:.3f})"

        return verdicts
    
    
    

    # -------------Core Aggregation -------------
    #Override evaluate aggregation to capture accuracy in real time.
    #feeding accuracy into the blame/reward logic
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[BaseException, Tuple[ClientProxy, fl.common.EvaluateRes]]],) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}
    
        # Weighted average accuracy across all clients
        total_examples = sum(r.num_examples for _, r in results)
        
        weighted_acc   = sum(
            r.num_examples * r.metrics["accuracy"]
            
            #r.num_clients * r.metrics["accuracy"]
            for _, r in results) / total_examples
    
        # Store values for next round
        self.prev_accuracy = (self.accuracy_history[-1] if self.accuracy_history else None)
        
        self.accuracy_history.append(weighted_acc)
    
        print(f"  Round {server_round} accuracy recorded: {weighted_acc:.4f}"
              f" | acc = {weighted_acc - self.prev_accuracy:.4f}"
              if self.prev_accuracy is not None
              else f"  Round {server_round} accuracy recorded: {weighted_acc:.4f} (baseline)")
    
        return weighted_acc, {"accuracy": weighted_acc}

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[BaseException, Tuple[ClientProxy, FitRes]]],) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        self.current_round = server_round

        if not results:
            return None, {}

        #  1. Extract client updates -------------
        client_data = []
        for i, (client_proxy, fit_res) in enumerate(results):
            
            #taking the results of client training, extracting the parameters and storing it into params for flattening earlier
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

        # 2. Retrospective update using LAST round's accuracy -------------
        # at the end of the previous round's evaluation
        current_acc = (
            self.accuracy_history[-1]
            if self.accuracy_history else None
        )

        if current_acc is not None:
            verdicts = self._retrospective_update(current_acc, client_ids)
        else:
            verdicts = {cid: "SKIP" for cid in client_ids}

        # 3. Compute global update direction -------------
        # reference for deviation scoring and weighted mean here
        all_flat      = np.array([cd["flat"] for cd in client_data])
        global_delta  = np.mean(all_flat, axis=0)

        # Store per-client deviations from global for next round's blame
        self.prev_client_deltas = {
            cd["id"]: cd["flat"] - global_delta
            for cd in client_data
        }
        self.prev_global_delta = global_delta

        # 4. Trust-weighted aggregation -------------
        aggregated  = None
        total_weight = 0.0
        
        #print and log the accumulation
        print(f"\n[ROUND {server_round}] ADL Trust Weights:")
        for cd in client_data:
            cid    = cd["id"]
            weight = self.trust_weights.get(cid, 1.0)
            verdict = verdicts.get(cid, "")

            print(f"  Client {cid:>2}: weight={weight:.3f}  {verdict}")

            #Log trust history
            self.trust_history[cid].append(round(weight, 4))

            total_weight += weight
            if aggregated is None:
                aggregated = [weight * layer for layer in cd["params"]]
            else:
                for j, layer in enumerate(cd["params"]):
                    aggregated[j] += weight * layer

        # Normalise
        aggregated = [w / total_weight for w in aggregated]

        print(f" Total weight: {total_weight:.3f} "
              f"| Avg weight: {total_weight/len(client_data):.3f}")

        return ndarrays_to_parameters(aggregated), {}


#Stores accuracy for  aggregate_fit succeeding round
    def update_accuracy(self, accuracy: float):

        self.prev_accuracy = (self.accuracy_history[-1] if self.accuracy_history else None)
        self.accuracy_history.append(accuracy)

    def get_trust_summary(self) -> Dict:
        return {
            "trust_weights_final": self.trust_weights,
            "trust_history":       self.trust_history,
            "accuracy_history":    self.accuracy_history,
            "exclusion_log":       self.exclusion_log,
        }