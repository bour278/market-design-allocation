from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

@dataclass(frozen=True)
class BidDecision:
    """Individual bid decision"""
    bid_id: int
    employer_id: int
    accept: bool
    timestamp: datetime
    bid_amount: float

@dataclass(frozen=True)
class Decision:
    """
    Complete decision/action representation for the allocation model.
    Represents the outcome of a policy decision, not the policy itself.
    """
    bid_decisions: List[BidDecision]
    total_accepted: int
    total_rejected: int
    
    @classmethod
    def create_empty(cls) -> 'Decision':
        """Create an empty decision state"""
        return cls(
            bid_decisions=[],
            total_accepted=0,
            total_rejected=0
        )
    
    @classmethod
    def from_binary_list(cls, 
                        decisions: List[bool],
                        bids: List[Dict],
                        timestamp: Optional[datetime] = None) -> 'Decision':
        """
        Create a Decision instance from a list of binary decisions
        Useful for policy implementations that output binary arrays
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        bid_decisions = [
            BidDecision(
                bid_id=idx,
                employer_id=bid['employer_id'],
                accept=decision,
                timestamp=timestamp,
                bid_amount=bid['amount']
            )
            for idx, (decision, bid) in enumerate(zip(decisions, bids))
        ]
        
        total_accepted = sum(1 for d in decisions if d)
        total_rejected = len(decisions) - total_accepted
        
        return cls(
            bid_decisions=bid_decisions,
            total_accepted=total_accepted,
            total_rejected=total_rejected
        )
    
    def get_accepted_bids(self) -> List[BidDecision]:
        """Get all accepted bids - useful for transition and objective functions"""
        return [d for d in self.bid_decisions if d.accept]
    
    def get_rejected_bids(self) -> List[BidDecision]:
        """Get all rejected bids - useful for transition and objective functions"""
        return [d for d in self.bid_decisions if not d.accept]
    
    def get_decision_for_employer(self, employer_id: int) -> Optional[BidDecision]:
        """Get decision for specific employer - useful for transition function"""
        for decision in self.bid_decisions:
            if decision.employer_id == employer_id:
                return decision
        return None
    
    def get_total_accepted_value(self) -> float:
        """Get total value of accepted bids - useful for objective function"""
        return sum(d.bid_amount for d in self.bid_decisions if d.accept)
    
    def get_acceptance_rate(self) -> float:
        """Get acceptance rate - useful for metrics"""
        if not self.bid_decisions:
            return 0.0
        return self.total_accepted / len(self.bid_decisions)
    
    def to_binary_list(self) -> List[bool]:
        """Convert to binary list - useful for policy implementations"""
        return [d.accept for d in self.bid_decisions]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary - useful for serialization"""
        return {
            'bid_decisions': [
                {
                    'bid_id': d.bid_id,
                    'employer_id': d.employer_id,
                    'accept': d.accept,
                    'timestamp': d.timestamp.isoformat(),
                    'bid_amount': d.bid_amount
                }
                for d in self.bid_decisions
            ],
            'total_accepted': self.total_accepted,
            'total_rejected': self.total_rejected
        } 