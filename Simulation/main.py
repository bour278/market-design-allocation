import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

from AllocationModel.state import State
from AllocationModel.decision import Decision
from AllocationModel.objective import ObjectiveFunction, ObjectiveTracker
from AllocationModel.transition import TransitionModel
from AllocationModel.exogenous import ExogenousInfoManager

from .simulation import RandomSimulation, SimulationConfig, SimulationDataSource

def run_random_simulation(config_path: Optional[str] = None):
    """Run simulation with random bidding behavior"""
    # Load config if provided
    if config_path:
        with open(config_path) as f:
            config_dict = json.load(f)
        config = SimulationConfig(**config_dict)
    else:
        config = SimulationConfig()
    
    simulation = RandomSimulation(config)
    data_source = SimulationDataSource(simulation)
    
    state = State.create_initial_state()
    transition_model = TransitionModel()
    objective = ObjectiveFunction()
    objective_tracker = ObjectiveTracker(objective, commission_rate=0.10)
    exogenous_manager = ExogenousInfoManager(data_source)
    
    n_steps = 100
    for step in range(n_steps):
        exogenous_info = exogenous_manager.get_info()
        
        print(f"\nStep {step}:")
        print(f"New bids: {len(exogenous_info.new_bids)}")
        print(f"New employers: {len(exogenous_info.market_update.new_employers)}")
        print(f"New workers: {len(exogenous_info.market_update.new_workers)}")
        
        current_auction = state.current_auction.add_bids(exogenous_info.new_bids)
        state = state.update(current_auction=current_auction)
        
        decision = transition_model.get_optimal_decision(state)
        
        next_state = transition_model.transition(state, decision, exogenous_info)
        
        objective_tracker.update(state, decision, next_state)
        
        state = next_state
    
    stats = objective_tracker.get_metric_statistics()
    print("\nSimulation Results:")
    for metric_name, metric_stats in stats.items():
        print(f"\n{metric_name}:")
        for stat_name, value in metric_stats.items():
            print(f"  {stat_name}: {value:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Run market allocation simulation')
    parser.add_argument('--mode', choices=['random'], default='random',
                      help='Simulation mode')
    parser.add_argument('--config', type=str,
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    if args.mode == 'random':
        run_random_simulation(args.config)
    else:
        raise ValueError(f"Unknown simulation mode: {args.mode}")

if __name__ == "__main__":
    main()
