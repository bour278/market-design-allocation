from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.integrate import quad
from .decision import Decision
from .state import State, MarketConditions, MarketState
from .exogenous import ExogenousInfo
import warnings

@dataclass
class TransitionHistory:
    """Stores the history of states, decisions, and rewards for learning"""
    states: List[State]
    decisions: List[Decision]
    rewards: List[float]
    next_states: List[State]

class BayesianOptimizer:
    """Implements Bayesian optimization with multiple acquisition functions"""
    def __init__(self, kernel=None):
        """Initialize optimizer with Gaussian Process and Matérn kernel"""
        if kernel is None:
            kernel = ConstantKernel(1.0) * Matern(
                length_scale=np.ones(1),
                nu=2.5
            )
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state=42
        )
        self.X_train = []
        self.y_train = []
        
    def entropy_search(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Computes entropy search acquisition function values for given points"""
        mu, sigma = self.gp.predict(X, return_std=True)
        samples = np.random.normal(
            loc=mu.reshape(-1, 1),
            scale=sigma.reshape(-1, 1),
            size=(len(X), n_samples)
        )
        
        def entropy_reduction(x_idx: int) -> float:
            """Calculates expected entropy reduction for a specific point"""
            current_entropy = self._calculate_max_entropy(samples)
            future_entropies = []
            
            for sample in samples[x_idx]:
                X_new = np.vstack([self.X_train, X[x_idx].reshape(1, -1)])
                y_new = np.append(self.y_train, sample)
                
                self.gp.fit(X_new, y_new)
                
                mu_new, sigma_new = self.gp.predict(X, return_std=True)
                samples_new = np.random.normal(
                    loc=mu_new.reshape(-1, 1),
                    scale=sigma_new.reshape(-1, 1),
                    size=(len(X), n_samples)
                )
                
                future_entropy = self._calculate_max_entropy(samples_new)
                future_entropies.append(future_entropy)
            
            expected_future_entropy = np.mean(future_entropies)
            return current_entropy - expected_future_entropy
        
        entropy_reductions = np.array([
            entropy_reduction(i) for i in range(len(X))
        ])
        
        return entropy_reductions

    def _calculate_max_entropy(self, samples: np.ndarray) -> float:
        """Calculates entropy of the maximum location distribution"""
        max_locations = np.argmax(samples, axis=0)
        unique, counts = np.unique(max_locations, return_counts=True)
        probs = counts / len(max_locations)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy

    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Computes expected improvement acquisition function values"""
        mu, sigma = self.gp.predict(X, return_std=True)
        mu_sample = self.gp.predict(self.X_train)
        
        sigma = sigma.reshape(-1, 1)
        mu_sample_opt = np.max(mu_sample)
        
        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
            
        return ei

    def upper_confidence_bound(self, X: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """Computes UCB acquisition function values"""
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu + beta * sigma.reshape(-1, 1)

class TransitionModel:
    """Manages state transitions and learning using Bayesian optimization"""
    def __init__(self):
        """Initialize transition model with GP optimizer"""
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        
        k1 = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
        k2 = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
        kernel = k1 * k2
        
        self.optimizer = BayesianOptimizer(kernel=kernel)
        
        self.history = TransitionHistory(
            states=[],
            decisions=[],
            rewards=[],
            next_states=[]
        )
        
        self.X_train = np.array([]).reshape(0, 5)
        self.y_train = np.array([])
        self.gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        self.acquisition_functions = {
            'ucb': self.upper_confidence_bound,
            'ei': self.expected_improvement,
            'pi': self.probability_improvement
        }
        
    def transition(self, current_state: State, decision: Decision, exogenous_info: ExogenousInfo) -> State:
        """Execute state transition with learning update"""
        features = self._extract_features(current_state, decision)
        new_state = self._update_market_state(current_state, decision, exogenous_info)
        reward = self._calculate_reward(current_state, decision, new_state)
        
        print(f"\nTransition Update:")
        print(f"Features: {features.flatten()}")
        print(f"Reward: {reward}")
        
        if not hasattr(self, 'X_train'):
            self.X_train = features
            self.y_train = np.array([reward])
        else:
            self.X_train = np.vstack([self.X_train, features])
            self.y_train = np.append(self.y_train, reward)
            
        print(f"Training data size: {len(self.y_train)}")
        print(f"Average reward: {np.mean(self.y_train):.3f}")
        
        # Fit GP model
        if len(self.y_train) > 1:
            self.optimizer.gp.fit(self.X_train, self.y_train)
        
        # Update history
        self.history.states.append(current_state)
        self.history.decisions.append(decision)
        self.history.next_states.append(new_state)
        
        return new_state

    def _extract_features(self, state: State, decision: Decision) -> np.ndarray:
        """Extract feature vector from state-action pair for GP"""
        features = [
            state.D_t.bid_spread,
            state.D_t.worker_to_job_ratio,
            state.S_t.market_liquidity,
            decision.get_acceptance_rate(),
            decision.get_total_accepted_value()
        ]
        return np.array(features).reshape(1, -1)

    def _calculate_reward(self, state: State, decision: Decision, next_state: State) -> float:
        """Calculate reward for a state-action-next_state transition"""
        match_reward = decision.get_total_accepted_value() * 0.4
        market_health = next_state.S_t.market_liquidity * next_state.D_t.worker_to_job_ratio * 0.4
        stability = (1 - abs(next_state.D_t.worker_to_job_ratio - 1.0)) * 0.2
        return match_reward + market_health + stability

    def _update_market_state(self, state: State, decision: Decision, exogenous_info: ExogenousInfo) -> State:
        """Update market state based on decision and exogenous information"""
        new_market_conditions = MarketConditions(
            bid_spread=self._calculate_new_spread(state, decision),
            n_available_workers=(
                state.D_t.n_available_workers +
                len(exogenous_info.market_update.new_workers) -
                len(exogenous_info.market_update.departed_workers)
            ),
            worker_to_job_ratio=self._calculate_new_ratio(state, exogenous_info)
        )
        
        new_market_state = MarketState(
            market_liquidity=state.S_t.market_liquidity,
            active_employers=(
                state.S_t.active_employers +
                len(exogenous_info.market_update.new_employers) -
                len(exogenous_info.market_update.departed_employers)
            ),
            employer_history=state.S_t.employer_history
        )
        
        return state.update(
            D_t=new_market_conditions,
            S_t=new_market_state
        )

    def get_optimal_decision(self, state: State, acquisition_function: str = 'ucb') -> Decision:
        print(f"\nDecision Debug:")
        print(f"Active bids: {len(state.current_auction.active_bids)}")
        if len(state.current_auction.active_bids) > 0:
            print(f"First bid amount: {state.current_auction.active_bids[0].amount}")
        
        if len(self.y_train) < 10:
            print("Building initial training data")
            n_bids = len(state.current_auction.active_bids)
            random_decisions = np.random.rand(n_bids) > 0.5
            bids = [
                {'employer_id': bid.employer_id, 'amount': bid.amount}
                for bid in state.current_auction.active_bids
            ]
            return Decision.from_binary_list(decisions=random_decisions.tolist(), bids=bids)
        
        candidates = self._generate_candidates(state)
        acq_values = self.acquisition_functions[acquisition_function](candidates)
        best_idx = np.argmax(acq_values)
        return self._candidate_to_decision(candidates[best_idx], state)

    def _generate_candidates(self, state: State) -> np.ndarray:
        """Generate candidate decisions for optimization"""
        n_candidates = 100
        n_bids = len(state.current_auction.active_bids)
        
        decisions = np.random.rand(n_candidates, n_bids)
        
        for i, bid in enumerate(state.current_auction.active_bids):
            quality_score = (bid.amount - state.D_t.base_bid_mean) / state.D_t.base_bid_std
            
            market_pressure = state.D_t.worker_to_job_ratio
            liquidity_factor = state.S_t.market_liquidity
            
            decisions[:, i] += 0.2 * quality_score  # Bid quality
            decisions[:, i] += 0.1 * market_pressure  # Market pressure
            decisions[:, i] += 0.1 * liquidity_factor  # Market liquidity
            
            if bid.amount < state.D_t.base_bid_mean * 0.8:
                decisions[:, i] -= 0.3
        
        decisions = (decisions > 0.6).astype(float)
        
        candidates = np.zeros((n_candidates, 5))
        for i in range(n_candidates):
            temp_decision = Decision.from_binary_list(
                decisions=decisions[i].tolist(),
                bids=[{'employer_id': bid.employer_id, 'amount': bid.amount}
                      for bid in state.current_auction.active_bids]
            )
            
            candidates[i] = [
                state.D_t.bid_spread,
                state.D_t.worker_to_job_ratio,
                state.S_t.market_liquidity,
                temp_decision.get_acceptance_rate(),
                temp_decision.get_total_accepted_value()
            ]
        
        return candidates

    def _candidate_to_decision(self, candidate: np.ndarray, state: State) -> Decision:
        """Convert optimization candidate back to Decision"""

        threshold = 0.5
        n_bids = len(state.current_auction.active_bids)
        decisions = (candidate[:n_bids] > threshold).astype(bool)
        
        bids = [
            {'employer_id': bid.employer_id, 'amount': bid.amount}
            for bid in state.current_auction.active_bids
        ]
        
        return Decision.from_binary_list(
            decisions=decisions.tolist(),
            bids=bids
        )

    def _calculate_new_spread(self, state: State, decision: Decision) -> float:
        """Calculate new bid spread based on accepted and rejected bids"""
        accepted_bids = decision.get_accepted_bids()
        if not accepted_bids:
            return state.D_t.bid_spread
        
        accepted_amounts = [bid.bid_amount for bid in accepted_bids]
        return max(accepted_amounts) - min(accepted_amounts)

    def _calculate_new_ratio(self, state: State, exogenous_info: ExogenousInfo) -> float:
        """Calculate new worker to job ratio"""
        new_employers = max(1, 
            state.S_t.active_employers +
            len(exogenous_info.market_update.new_employers) -
            len(exogenous_info.market_update.departed_employers)
        )
        
        new_workers = max(1,
            state.D_t.n_available_workers +
            len(exogenous_info.market_update.new_workers) -
            len(exogenous_info.market_update.departed_workers)
        )
        
        return new_workers / new_employers 

    def upper_confidence_bound(self, X: np.ndarray, kappa: float = 2.0) -> np.ndarray:
        """UCB acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu + kappa * sigma

    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """EI acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        mu_sample = self.gp.predict(self.X_train)
        
        mu_sample_opt = np.max(mu_sample)
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

    def probability_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """PI acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        mu_sample = self.gp.predict(self.X_train)
        
        mu_sample_opt = np.max(mu_sample)
        Z = (mu - mu_sample_opt - xi) / sigma
        return norm.cdf(Z) 