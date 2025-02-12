from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from AllocationModel.exogenous import NewBid, MarketUpdate, DataSource
from AllocationModel.state import State, MarketConditions, MarketState, CurrentAuction

@dataclass
class SimulationConfig:
    """Configuration for random simulation"""
    n_employers: int = 100
    n_workers: int = 200
    base_bid_mean: float = 100.0
    base_bid_std: float = 20.0
    urgency_alpha: float = 2.0
    urgency_beta: float = 5.0
    employer_arrival_rate: float = 0.1
    employer_departure_rate: float = 0.05
    worker_arrival_rate: float = 0.15
    worker_departure_rate: float = 0.08
    time_step: int = 1

class RandomSimulation:
    """Simple simulation with random distributions"""
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.current_time = datetime.now()
        self.next_employer_id = self.config.n_employers
        self.active_employers = set(range(self.config.n_employers))
        self.active_workers = set(range(self.config.n_workers))
        self.n_available_workers = self.config.n_workers
        self.rng = np.random.default_rng(42)
        
    def generate_bids(self) -> List[NewBid]:
        """Generate new bids from random employers"""
        n_new_bids = self.rng.poisson(len(self.active_employers) * 0.2)
        
        new_bids = []
        for _ in range(n_new_bids):
            employer_id = self.rng.choice(list(self.active_employers))
            
            amount = self.rng.normal(
                self.config.base_bid_mean,
                self.config.base_bid_std
            )
            amount = max(0.1, amount)
            
            urgency = self.rng.beta(
                self.config.urgency_alpha,
                self.config.urgency_beta
            )
            
            new_bids.append(NewBid(
                employer_id=employer_id,
                amount=amount,
                timestamp=self.current_time,
                job_id=self.rng.integers(0, 1000000),
                urgency=urgency
            ))
            
        return new_bids
    
    def get_market_update(self) -> MarketUpdate:
        """Generate market update with employer/worker changes"""
        n_new_employers = self.rng.poisson(
            self.config.employer_arrival_rate * 
            (100 if len(self.active_employers) < 10 else 1)
        )
        
        new_employers = list(range(
            self.next_employer_id,
            self.next_employer_id + n_new_employers
        ))
        self.next_employer_id += n_new_employers
        
        self.active_employers.update(new_employers)
        
        n_departures = self.rng.poisson(
            len(self.active_employers) * self.config.employer_departure_rate
        )
        if n_departures > 0 and len(self.active_employers) > n_departures:
            departed_employers = self.rng.choice(
                list(self.active_employers),
                size=n_departures,
                replace=False
            ).tolist()
            self.active_employers.difference_update(departed_employers)
        else:
            departed_employers = []
        
        n_new_workers = self.rng.poisson(self.config.worker_arrival_rate * 100)
        n_departed_workers = self.rng.poisson(
            self.config.worker_departure_rate * 
            self.n_available_workers
        )
        
        new_workers = list(range(n_new_workers))
        departed_workers = list(range(n_departed_workers))
        
        self.n_available_workers += (n_new_workers - n_departed_workers)
        self.n_available_workers = max(1, self.n_available_workers)
        
        return MarketUpdate(
            new_employers=new_employers,
            departed_employers=departed_employers,
            new_workers=new_workers,
            departed_workers=departed_workers,
            timestamp=self.current_time
        )
    
    def step(self):
        """Advance simulation time"""
        self.current_time += timedelta(minutes=self.config.time_step)

class SimulationDataSource(DataSource):
    """Data source adapter for simulation"""
    def __init__(self, simulation: RandomSimulation):
        self.simulation = simulation
        self.current_time = datetime.now()
    
    def get_new_bids(self) -> List[NewBid]:
        """Get new bids from simulation"""
        bids = self.simulation.generate_bids()
        self.current_time += timedelta(minutes=1)
        self.simulation.current_time = self.current_time
        return bids
    
    def get_market_update(self) -> MarketUpdate:
        """Get market update from simulation"""
        update = self.simulation.get_market_update()
        self.current_time += timedelta(minutes=1)
        self.simulation.current_time = self.current_time
        return update 