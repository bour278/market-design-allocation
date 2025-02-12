from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import asdict

@dataclass(frozen=True)
class FairValue:
    """Fair value estimates"""
    employer_id: int
    worker_reservation: float
    market_value: float
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.worker_reservation <= 0:
            raise ValueError("Worker reservation must be positive")
        if self.market_value < self.worker_reservation:
            raise ValueError("Market value cannot be less than worker reservation")

@dataclass(frozen=True)
class BidderProfile:
    current_bid: float
    estimated_max_bid: Optional[float]
    historical_max: float
    bid_flexibility: float  # Between 0 and 1

    def __post_init__(self):
        if self.current_bid <= 0:
            raise ValueError("Current bid must be positive")
        if self.bid_flexibility < 0 or self.bid_flexibility > 1:
            raise ValueError("Bid flexibility must be between 0 and 1")

@dataclass(frozen=True)
class MarketConditions:
    """Current market conditions"""
    bid_spread: float
    worker_to_job_ratio: float
    n_available_workers: int
    base_bid_mean: float = 100.0
    base_bid_std: float = 20.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.n_available_workers < 0:
            raise ValueError("Number of workers cannot be negative")
        if self.worker_to_job_ratio <= 0:
            raise ValueError("Worker to job ratio must be positive")

@dataclass(frozen=True)
class JobCharacteristics:
    duration: int
    urgency: float      # Between 0 and 1
    flexibility: float  # Between 0 and 1

@dataclass(frozen=True)
class EmployerProfile:
    """Employer bidding profile"""
    employer_id: int
    participation_rate: float
    win_rate: float
    avg_bid: float
    last_active: datetime

@dataclass(frozen=True)
class Bid:
    employer_id: int
    amount: float
    timestamp: datetime

@dataclass(frozen=True)
class MarketState:
    """Historical market state"""
    market_liquidity: float
    active_employers: int
    employer_history: Dict[int, EmployerProfile] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass(frozen=True)
class CurrentAuction:
    """Current auction state"""
    active_bids: List[int] = field(default_factory=list)
    fair_values: Dict[int, FairValue] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def add_bids(self, new_bids: List[int]) -> 'CurrentAuction':
        """Add new bids to current auction"""
        return CurrentAuction(
            active_bids=self.active_bids + new_bids
        )

@dataclass(frozen=True)
class DecisionFactors:
    market_conditions: MarketConditions
    fair_values: Dict[int, FairValue] = field(default_factory=dict)
    bidder_profiles: Dict[int, BidderProfile] = field(default_factory=dict)
    job_characteristics: Dict[int, JobCharacteristics] = field(default_factory=dict)

@dataclass(frozen=True)
class SupportingState:
    market_state: MarketState
    employer_history: Dict[int, EmployerProfile] = field(default_factory=dict)
    current_auction: CurrentAuction = field(default_factory=CurrentAuction)

@dataclass(frozen=True)
class State:
    """Complete system state"""
    D_t: MarketConditions
    S_t: MarketState
    current_auction: CurrentAuction = field(default_factory=CurrentAuction)
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def create_initial_state(cls) -> 'State':
        """Create initial state with default values"""
        initial_market_conditions = MarketConditions(
            bid_spread=0.0,
            worker_to_job_ratio=1.0,
            n_available_workers=100,
            base_bid_mean=100.0,
            base_bid_std=20.0
        )
        
        initial_market_state = MarketState(
            market_liquidity=1.0,
            active_employers=0
        )
        
        initial_auction = CurrentAuction()
        
        return cls(
            D_t=initial_market_conditions,
            S_t=initial_market_state,
            current_auction=initial_auction
        )

    def update(self, **kwargs) -> 'State':
        """Create new state with updated values"""
        current_values = {
            'D_t': self.D_t,
            'S_t': self.S_t,
            'current_auction': self.current_auction,
            'timestamp': self.timestamp
        }
        current_values.update(kwargs)
        return State(**current_values) 