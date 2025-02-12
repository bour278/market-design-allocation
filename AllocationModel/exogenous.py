from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Any
from datetime import datetime

@dataclass(frozen=True)
class NewBid:
    employer_id: int
    amount: float
    timestamp: datetime
    job_id: int
    urgency: float

@dataclass(frozen=True)
class MarketUpdate:
    new_employers: List[int]
    departed_employers: List[int]
    new_workers: List[int]
    departed_workers: List[int]
    timestamp: datetime

@dataclass(frozen=True)
class ExogenousInfo:
    """
    Container for all exogenous information arriving after a decision
    """
    new_bids: List[NewBid]
    market_update: MarketUpdate
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def create_empty(cls) -> 'ExogenousInfo':
        """Create empty exogenous information"""
        empty_market_update = MarketUpdate(
            new_employers=[],
            departed_employers=[],
            new_workers=[],
            departed_workers=[],
            timestamp=datetime.now()
        )
        return cls(
            new_bids=[],
            market_update=empty_market_update,
            timestamp=datetime.now()
        )

class DataSource(Protocol):
    """Protocol for data sources (simulation or real)"""
    def get_new_bids(self) -> List[NewBid]:
        """Get new bids from the source"""
        pass

    def get_market_update(self) -> MarketUpdate:
        """Get market updates from the source"""
        pass

class ExogenousInfoManager:
    """
    Manager class to handle exogenous information from different sources
    """
    def __init__(self, data_source: DataSource):
        self.data_source = data_source
        self._last_update: Optional[datetime] = None

    def get_info(self) -> ExogenousInfo:
        """Get new exogenous information from the data source"""
        # Get new information from data source
        new_bids = self.data_source.get_new_bids()
        market_update = self.data_source.get_market_update()
        
        current_time = datetime.now()
        self._last_update = current_time
        
        # Process and validate the information
        if new_bids is None:
            new_bids = []
        if market_update is None:
            market_update = MarketUpdate(
                new_employers=[],
                departed_employers=[],
                new_workers=[],
                departed_workers=[],
                timestamp=current_time
            )
        
        # Create and return ExogenousInfo
        return ExogenousInfo(
            new_bids=new_bids,
            market_update=market_update,
            timestamp=current_time
        )

class SimulatedDataSource(DataSource):
    """Example implementation for simulation"""
    def __init__(self, simulation_engine: Any):
        self.simulation = simulation_engine
        
    def get_new_bids(self) -> List[NewBid]:
        return self.simulation.generate_bids()
        
    def get_market_update(self) -> MarketUpdate:
        return self.simulation.get_market_changes()

class HistoricalDataSource(DataSource):
    """Example implementation for historical data"""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.current_index = 0
        
    def get_new_bids(self) -> List[NewBid]:
        raise NotImplementedError("Historical data source not implemented yet")
        
    def get_market_update(self) -> MarketUpdate:
        raise NotImplementedError("Historical data source not implemented yet")

class LiveDataSource(DataSource):
    """Example implementation for live data"""
    def __init__(self, api_client: Any):
        self.api_client = api_client
        
    def get_new_bids(self) -> List[NewBid]:
        raise NotImplementedError("Live data source not implemented yet")
        
    def get_market_update(self) -> MarketUpdate:
        raise NotImplementedError("Live data source not implemented yet")

@dataclass
class ExogenousInfoBuffer:
    """Buffer to store and manage exogenous information"""
    max_size: int = 1000
    _buffer: List[ExogenousInfo] = field(default_factory=list)
    
    def add(self, info: ExogenousInfo):
        """Add new information to buffer"""
        self._buffer.append(info)
        if len(self._buffer) > self.max_size:
            self._buffer.pop(0)
    
    def get_recent(self, n: int = 1) -> List[ExogenousInfo]:
        """Get n most recent information entries"""
        return self._buffer[-n:]
    
    def clear(self):
        """Clear the buffer"""
        self._buffer.clear() 