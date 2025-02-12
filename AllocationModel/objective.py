from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import numpy as np

from .state import State
from .decision import Decision

@dataclass
class Metric:
    """Defines a single optimization metric"""
    name: str
    weight: float
    is_minimization: bool
    calculation: Callable[[State, Decision, State], float]
    threshold: Optional[float] = None

class ObjectiveFunction:
    """
    Min-max objective function for market optimization.
    Balances multiple competing objectives in a non-convex setting.
    """
    def __init__(self):
        self.maximize_metrics = {
            'platform_profits': Metric(
                name='platform_profits',
                weight=0.4,
                is_minimization=False,
                calculation=lambda s, d, ns: d.get_total_accepted_value()
            ),
            'match_rate': Metric(
                name='match_rate',
                weight=0.3,
                is_minimization=False,
                calculation=lambda s, d, ns: d.get_acceptance_rate()
            ),
            'market_liquidity': Metric(
                name='market_liquidity',
                weight=0.2,
                is_minimization=False,
                calculation=lambda s, d, ns: ns.S_t.market_liquidity
            )
        }
        
        self.minimize_metrics = {
            'leave_rate': Metric(
                name='leave_rate',
                weight=0.1,
                is_minimization=True,
                calculation=lambda s, d, ns: len([
                    emp for emp in ns.S_t.employer_history.values()
                    if emp.participation_rate < 0.2
                ]) / max(1, len(ns.S_t.employer_history))
            )
        }
        
        self.satisfaction_metrics = {
            'bidder_satisfaction': Metric(
                name='bidder_satisfaction',
                weight=0.1,
                is_minimization=False,
                calculation=lambda s, d, ns: np.mean([
                    emp.win_rate for emp in ns.S_t.employer_history.values()
                ])
            ),
            'worker_satisfaction': Metric(
                name='worker_satisfaction',
                weight=0.1,
                is_minimization=False,
                calculation=lambda s, d, ns: np.mean([
                    1 if bid.bid_amount >= s.D_t.worker_to_job_ratio * bid.bid_amount
                    else 0 for bid in d.bid_decisions if bid.accept
                ])
            )
        }

    def __call__(self, 
                 current_state: State, 
                 decision: Decision, 
                 next_state: State) -> float:
        """
        Calculate composite min-max objective value.
        Higher return value indicates better overall performance.
        """
        max_objectives = self._calculate_maximization_objectives(
            current_state, decision, next_state
        )
        min_objectives = self._calculate_minimization_objectives(
            current_state, decision, next_state
        )
        satisfaction = self._calculate_satisfaction_score(
            current_state, decision, next_state
        )
        
        return max_objectives - min_objectives + satisfaction

    def _calculate_maximization_objectives(self, 
                                        current_state: State, 
                                        decision: Decision, 
                                        next_state: State) -> float:
        values = []
        for metric in self.maximize_metrics.values():
            value = metric.calculation(current_state, decision, next_state)
            normalized = self._normalize_metric(metric.name, value)
            values.append(normalized * metric.weight)
        return np.sum(values)

    def _calculate_minimization_objectives(self, 
                                        current_state: State, 
                                        decision: Decision, 
                                        next_state: State) -> float:
        values = []
        for metric in self.minimize_metrics.values():
            value = metric.calculation(current_state, decision, next_state)
            normalized = self._normalize_metric(metric.name, value)
            values.append(normalized * metric.weight)
        return np.sum(values)

    def _calculate_satisfaction_score(self, 
                                   current_state: State, 
                                   decision: Decision, 
                                   next_state: State) -> float:
        values = []
        for metric in self.satisfaction_metrics.values():
            value = metric.calculation(current_state, decision, next_state)
            normalized = self._normalize_metric(metric.name, value)
            values.append(normalized * metric.weight)
        return np.sum(values)

    def _normalize_metric(self, metric_name: str, value: float) -> float:
        """Normalize metric value to [0,1] range"""
        all_metrics = {
            **self.maximize_metrics,
            **self.minimize_metrics,
            **self.satisfaction_metrics
        }
        metric = all_metrics[metric_name]
        
        if metric.threshold is not None:
            return min(1.0, value / metric.threshold)
        return value

class ObjectiveTracker:
    """Tracks objective function values over time"""
    def __init__(self, objective_function: ObjectiveFunction, commission_rate: float = 0.10):
        self.objective_function = objective_function
        self.commission_rate = commission_rate
        self.history = []
        
    def update(self, current_state: State, decision: Decision, next_state: State):
        """Update tracker with new state transition"""
        value = self.objective_function(
            current_state,
            decision,
            next_state
        )
        self.history.append({
            'timestamp': next_state.timestamp,
            'value': value,
            'state': next_state,
            'decision': decision
        })
        
    def get_metric_statistics(self, window: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for tracked metrics"""
        if not self.history:
            return {}
            
        if window is not None:
            history = self.history[-window:]
        else:
            history = self.history
            
        # Debug prints
        print("\nDebug Info:")
        print(f"History length: {len(history)}")
        print(f"First decision total accepted: {history[0]['decision'].get_total_accepted_value()}")
        print(f"First decision acceptance rate: {history[0]['decision'].get_acceptance_rate()}")
        
        # Calculate platform profits with commission
        profits = [d['decision'].get_total_accepted_value() * self.commission_rate for d in history]
        match_rates = [d['decision'].get_acceptance_rate() for d in history]
        
        # More debug prints
        print(f"Average profit before stats: {np.mean(profits) if profits else 0}")
        print(f"Average match rate before stats: {np.mean(match_rates) if match_rates else 0}")
        
        stats = {
            'platform_profits': {
                'mean': float(np.mean(profits)) if profits else 0.0,
                'std': float(np.std(profits)) if profits else 0.0,
                'min': float(np.min(profits)) if profits else 0.0,
                'max': float(np.max(profits)) if profits else 0.0
            },
            'match_rate': {
                'mean': float(np.mean(match_rates)) if match_rates else 0.0,
                'std': float(np.std(match_rates)) if match_rates else 0.0,
                'min': float(np.min(match_rates)) if match_rates else 0.0,
                'max': float(np.max(match_rates)) if match_rates else 0.0
            },
            'market_health': {
                'mean': float(np.mean([d['state'].S_t.market_liquidity * d['state'].D_t.worker_to_job_ratio for d in history])),
                'std': float(np.std([d['state'].S_t.market_liquidity * d['state'].D_t.worker_to_job_ratio for d in history])),
                'min': float(np.min([d['state'].S_t.market_liquidity * d['state'].D_t.worker_to_job_ratio for d in history])),
                'max': float(np.max([d['state'].S_t.market_liquidity * d['state'].D_t.worker_to_job_ratio for d in history]))
            }
        }
        
        return stats 