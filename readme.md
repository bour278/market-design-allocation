# Market Design Allocation System

## Framework Components

1. **State ($S_t$)**: Represents the complete system state including:
   - Market conditions
   - Bidder profiles
   - Historical data
   - Current auction state

2. **Decision ($X_t$)**: Represents actions taken in response to the state:
   - Bid acceptance/rejection decisions
   - Market clearing decisions

3. **Exogenous Information ($W_{t+1}$)**: External information arriving after decisions:
   - New bids
   - Market participant arrivals/departures
   - Market condition changes

4. **Transition Function ($S^M$)**: State evolution implementing Bayesian optimization:
   - Updates state based on decisions and exogenous information
   - Learns optimal decisions through GP surrogate models
   - Implements various acquisition functions

5. **Objective Function**: Multi-objective optimization considering:
   - Platform profits
   - Match rates
   - Market liquidity
   - Participant satisfaction

## Bayesian Optimization

The system uses Bayesian optimization with Gaussian Process surrogate models to learn optimal decisions. The key components are:

### Gaussian Process Prior
We place a GP prior on the objective function $f$:

$$f(x) \sim \mathcal{GP}(\mu(x), k(x,x'))$$

where $\mu(x)$ is the mean function and $k(x,x')$ is the Matérn kernel with $\nu=2.5$.

### Acquisition Functions

1. **Expected Improvement (EI)**:
   
   $$EI(x) = \mathbb{E}[\max(0, f(x) - f(x^+))]$$
   
   $$= (\mu(x) - f(x^+))\Phi(Z) + \sigma(x)\phi(Z)$$

   where $Z = \frac{\mu(x) - f(x^+)}{\sigma(x)}$

2. **Upper Confidence Bound (UCB)**:
   
   $$UCB(x) = \mu(x) + \beta\sigma(x)$$

3. **Entropy Search**:
   
   $$ES(x) = H(x^*|D_n) - \mathbb{E}_{y|D_n,x}[H(x^*|D_n \cup \{(x,y)\})]$$

   where $H(x^*)$ is the entropy of the distribution over the location of the global maximum.

## Installation

Clone repository:

## Installation

Clone repository:

    git clone https://github.com/bour278/market-design-allocation.git
    cd market-design-allocation

Create virtual environment:

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Install requirements:

    pip install -r requirements.txt

## Running Simulations

### Random Simulation

    python -m Simulation.main --mode random

### With Custom Configuration

    python -m Simulation.main --mode random --config path/to/config.json

Example configuration:

    {
        "n_employers": 100,
        "n_workers": 200,
        "base_bid_mean": 100.0,
        "base_bid_std": 20.0,
        "urgency_alpha": 2.0,
        "urgency_beta": 5.0,
        "employer_arrival_rate": 0.1,
        "employer_departure_rate": 0.05,
        "worker_arrival_rate": 0.15,
        "worker_departure_rate": 0.08,
        "time_step": 1
    }

## Project Structure

Our implementation follows the five-component framework with the following structure:

    market-design-allocation/
    ├── AllocationModel/
    │   ├── __init__.py
    │   ├── state.py          # State representation (St)
    │   ├── decision.py       # Decision space (Xt)
    │   ├── exogenous.py      # Exogenous information (Wt+1)
    │   ├── transition.py     # Transition function with Bayesian optimization
    │   └── objective.py      # Multi-objective function
    ├── Simulation/
    │   ├── __init__.py
    │   ├── simulation.py     # Random simulation environment
    │   └── main.py          # Entry point and CLI
    ├── requirements.txt
    └── README.md

## Directory Structure Details

### AllocationModel/
- `state.py`: Implements market state including conditions, profiles, and history
- `decision.py`: Defines decision space for bid acceptance/rejection
- `exogenous.py`: Handles external information like new bids and market changes
- `transition.py`: Implements Bayesian optimization with GP surrogate models
- `objective.py`: Defines multi-objective optimization metrics

### Simulation/
- `simulation.py`: Implements random simulation with configurable distributions
- `main.py`: CLI interface and simulation runner
