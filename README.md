# Option Package Optimization Program

## Overview

This program helps optimize execution costs when trading options by grouping small notional trades into larger packages while respecting strike distance constraints. It implements different groupings and transaction cost models to analyze potential savings.

## Features

- Groups options within configurable strike distance constraints
- Compares two transaction cost models (tiered and logarithmic)
- Analyzes savings across different distance constraints
- Provides detailed trading instructions for execution
- Visualizes savings patterns by set, cost model, and distance

## Requirements

- Python 3.8 or higher
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - pickle
  - hashlib

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/option-package-optimization.git

# Install required packages
pip install pandas numpy matplotlib
```

## Usage

1. Place your option data in a CSV file named `products_to_package.csv` in the same directory as the script
2. Run the script:

```bash
python package-products.py
```

## Input Format

The input CSV file (`products_to_package.csv`) must contain the following columns:

- `set`: Integer identifier for option sets (options within the same set are candidates for grouping)
- `strike`: Option strike price
- `notional`: Notional amount (dollar value)

Example:

```
set,strike,notional
0,1.0035,1660000.0
0,1.0037,3480000.0
0,1.0054,450000.0
...
```

## Output Files

The script generates several output files:

| Filename | Description |
|----------|-------------|
| `package-products.csv` | CSV file with detailed results for each scenario |
| `package-products.csv.txt` | Text file with column definitions for the CSV |
| `package-products.pkl` | Pickle file with raw results data for further analysis |
| `trading_instructions.txt` | Text file with detailed trading instructions for all scenarios |
| `package-products.png` | Scatter plot of savings vs max distance |

## Analysis Metrics

- **original_trades**: Number of individual option trades before grouping
- **grouped_trades**: Number of consolidated trades after grouping
- **original_cost**: Total transaction cost if executing all trades individually
- **grouped_cost**: Total transaction cost after grouping trades
- **savings**: Absolute cost reduction from grouping
- **savings_percentage**: Percentage cost reduction from grouping
- **set_id**: Identifier for the option set
- **max_distance**: Maximum allowed distance between strikes for grouping
- **cost_model**: Transaction cost model used (simple or complex)

## Transaction Cost Models

1. **Simple (Tiered) Model**:
   - Trades ≤ $1M: 0.06% of notional
   - Trades > $1M: 0.03% of notional

2. **Complex (Logarithmic) Model**:
   - Cost = notional × (11 - 0.5 × ln(notional)) / 10000

## Maximum Distance Options

The script evaluates four maximum distance constraints:
- 0.02: Tight grouping, minimizes basis risk
- 0.03: Medium grouping
- 0.04: Looser grouping
- None: Unrestricted grouping, maximizes cost savings

## License

All rights reserved. Copyright (c) 2025 Lana A. Cartailler

## Contact

For questions or issues, please contact:  
Lana A. Cartailler https://www.linkedin.com/in/lana-cartailler/ 
