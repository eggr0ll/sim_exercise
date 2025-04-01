#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Lana A. Cartailler
# All rights reserved.

"""
File name: 
Author: Lana A. Cartailler
Created: February 24, 2025
Version: 1.0
Description: This program implements an options package optimization and grouping 
   algorithm. Its purpose is to group small notional options trades into 
   larger packages to reduce transaction costs while respecting strike distance 
   constraints.
License: All rights reserved. Copyright (c) 2025 Lana A. Cartailler
Contact: Lana A. Cartailler https://www.linkedin.com/in/lana-cartailler/
Dependencies: Python 3.8 or higher, pandas, numpy, matplotlib, pickle, io, sys
"""

# #######################################################
# DEPENDENCIES
#######################################################

import sys
import io
import pandas as pd
import numpy as np
import pickle
import hashlib
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for better handling of files
import matplotlib.pyplot as plt

# #########################################################
# READ AND VALIDATE DATA
# ########################################################
def read_validate_csv(file_path):
    """
    Validates and cleans options data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Cleaned and validated dataframe
    """
    # Read the CSV file
    try:
        df = pd.read_csv(file_path, skip_blank_lines=True)
    except Exception as e:
        raise Exception(f"Error reading file: {e}")
    
    # Check for required columns
    required_columns = ['set', 'strike', 'notional']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty values
    na_counts = df[required_columns].isna().sum()
    if na_counts.sum() > 0:
        print(f"WARNING: Found {na_counts.sum()} NA values: {na_counts.to_dict()}")
        df = df.dropna(subset=required_columns)
    
    # Convert data types
    try:
        df['set'] = df['set'].astype(int)
        df['strike'] = df['strike'].astype(float)
        df['notional'] = df['notional'].astype(float)
    except Exception as e:
        raise ValueError(f"Data type conversion error: {e}")
    
    # Basic validation checks
    if df['notional'].min() <= 0:
        print("WARNING: Found non-positive notional values")
    
    # Print summary
    print(f"Processed {len(df)} rows across {df['set'].nunique()} sets")
    print(f"Strike range: {df['strike'].min():.4f} to {df['strike'].max():.4f}")
    print(f"Notional range: ${df['notional'].min():,.2f} to ${df['notional'].max():,.2f}")
    
    return df


#######################################################
# TRANSACTION COST MODELS
#######################################################

def calculate_transaction_cost_simple(notional):
    """
    Implements the simple tiered transaction cost model specified in requirement:
    'Anything at/below $1m costs 0.06% per amount notional, anything above $1m costs 0.03%'
    
    This model reflects real market structure where:
    - Small trades (<$1M) face retail-like spreads and higher costs
    - Large trades (>$1M) get institutional pricing
    - Binary threshold represents typical market maker size brackets
    
    Args:
        notional: Dollar amount of the option trade
    Returns:
        Dollar transaction cost based on size tier
    """
    if notional <= 1000000:
        return notional * 0.0006  # 0.06% for retail-size trades
    return notional * 0.0003     # 0.03% for institutional-size trades

def calculate_transaction_cost_complex(notional):
    """
    Implements the continuous logarithmic cost model specified in requirement:
    '( 11 - 1/2 * ln(notional) ) / 10000'
    
    This sophisticated model captures:
    - Continuous improvement in execution costs with size
    - Diminishing returns via logarithmic function
    - Real market microstructure effects:
        * Market impact
        * Dealer attention
        * Spread compression for size
        * Price improvement for larger trades
    
    Args:
        notional: Dollar amount of the option trade
    Returns:
        Dollar transaction cost based on logarithmic scaling
    """
    return notional * (11 - 0.5 * np.log(notional)) / 10000

#######################################################
# OPTION GROUPING LOGIC
#######################################################

def group_options(data, max_distance, cost_function):
    """
    Core grouping algorithm that implements the main requirement:
    "The desired goal is to group over several strikes so that we can purchase 
    a single position with a larger notional at one strike"
    
    Key Features:
    1. Respects max_distance constraint for risk management
    2. Uses notional-weighted average for group strikes
    3. Processes each set independently
    4. Tracks both grouped results and original trades
    
    Risk Management:
    - max_distance controls deviation from original strikes
    - Weighted average strike preserves overall position delta
    - Grouping affects gamma and vega profiles
    
    Market Execution:
    - Combines small trades into larger blocks
    - Preserves execution audit trail
    - Enables transaction cost analysis
    
    Args:
        data: DataFrame with [set, strike, notional] columns
        max_distance: Strike distance constraint [0.02, 0.03, 0.04, None]
        cost_function: Transaction cost model to use
    
    Returns:
        tuple: (grouped trades DataFrame, grouping details list)
    """
    grouped_data = []
    grouping_details = []  # Track grouping details for analysis
    
    for set_id in data['set'].unique():
        set_data = data[data['set'] == set_id].sort_values('strike')
        
        # Initialize grouping
        current_group = []
        for _, row in set_data.iterrows():
            if not current_group:
                current_group.append(row)
                continue
                
            # Calculate notional-weighted average strike for the group
            # This preserves the average exposure point of the original trades
            base_strike = np.average([g['strike'] for g in current_group],
                                   weights=[g['notional'] for g in current_group])
                                   
            # Check max_distance constraint for risk management
            if max_distance is None or abs(row['strike'] - base_strike) <= max_distance:
                current_group.append(row)
            else:
                # Process and store current group
                total_notional = sum(g['notional'] for g in current_group)
                group_details = {
                    'original_strikes': [g['strike'] for g in current_group],
                    'original_notionals': [g['notional'] for g in current_group],
                }
                
                grouped_data.append({
                    'set': set_id,
                    'original_strikes': [g['strike'] for g in current_group],
                    'group_strike': base_strike,
                    'total_notional': total_notional,
                    'transaction_cost': cost_function(total_notional)
                })
                grouping_details.append(group_details)
                current_group = [row]
                
        # Process final group
        if current_group:
            total_notional = sum(g['notional'] for g in current_group)
            base_strike = np.average([g['strike'] for g in current_group],
                                   weights=[g['notional'] for g in current_group])
            
            group_details = {
                'original_strikes': [g['strike'] for g in current_group],
                'original_notionals': [g['notional'] for g in current_group],
            }
            
            grouped_data.append({
                'set': set_id,
                'original_strikes': [g['strike'] for g in current_group],
                'group_strike': base_strike,
                'total_notional': total_notional,
                'transaction_cost': cost_function(total_notional)
            })
            grouping_details.append(group_details)
    
    return pd.DataFrame(grouped_data), grouping_details

#######################################################
# TRADE EXECUTION AND ANALYSIS FUNCTIONS
#######################################################

def show_trading_instructions(data, grouped_data):
    """
    Generates executable trading instructions from the grouping results.
    
    Purpose (from problem description):
    "The desired goal is to group over several strikes so that we can purchase 
    a single position with a larger notional at one strike"
    
    Trading Implementation:
    - Shows exact strikes and notionals for execution
    - Maps grouped trades back to original positions
    - Provides audit trail for compliance
    - Enables pre-trade analysis
    
    Output Format:
    - Organized by set_id for portfolio management
    - Shows both new trades and replaced positions
    - Includes notional and strike details for risk analysis
    """
    print("\nTRADING INSTRUCTIONS:")
    print("--------------------")
    for set_id in sorted(grouped_data['set'].unique()):
        print(f"\nSET {set_id}:")
        set_groups = grouped_data[grouped_data['set'] == set_id]
        
        for idx, group in set_groups.iterrows():
            print(f"\nEXECUTE:")
            print(f"  BUY {group['total_notional']:,.2f} notional at strike {group['group_strike']:.4f}")
            print(f"  This replaces original trades:")
            original_trades = []
            for strike in group['original_strikes']:
                original_trade = data[(data['set'] == set_id) & 
                                    (data['strike'] == strike)].iloc[0]
                original_trades.append(original_trade)
            
            # Sort by strike for clearer output
            for trade in sorted(original_trades, key=lambda x: x['strike']):
                print(f"    - {trade['notional']:,.2f} at strike {trade['strike']:.4f}")

def evaluate_grouping(original_data, grouped_data, cost_function):
    """
    Evaluates the effectiveness of trade grouping by comparing costs.
    
    Purpose (from problem description):
    "evaluate the performance of the grouping algorithm" and
    "Show the impact of the maximum distance choice as well as the transaction cost models"
    
    Evaluation Metrics:
    - Number of trades reduction
    - Absolute cost savings
    - Percentage cost savings
    - Transaction cost comparison
    
    Market Efficiency Analysis:
    - Measures execution cost improvement
    - Quantifies operational efficiency gains
    - Provides cost-benefit analysis for grouping
    """
    
    individual_costs = []
    for _, row in original_data.iterrows():
        cost = cost_function(row['notional'])
        individual_costs.append((row['notional'], cost))
    
    """
    # Debug: print individual costs
    # Sort and print top 5 largest trades
    sorted_costs = sorted(individual_costs, key=lambda x: x[0], reverse=True)
    print("Top 5 largest trades:")
    for notional, cost in sorted_costs[:5]:
        print(f"Notional: ${notional:,.2f}, Cost: ${cost:,.2f}") 
        """
    
    original_cost = sum(cost for _, cost in individual_costs)
    grouped_cost = sum(row['transaction_cost'] for _, row in grouped_data.iterrows())
    
    return {
        'original_trades': len(original_data),
        'grouped_trades': len(grouped_data),
        'original_cost': original_cost,
        'grouped_cost': grouped_cost,
        'savings': original_cost - grouped_cost,
        'savings_percentage': (original_cost - grouped_cost) / original_cost * 100
    }

def create_direct_visualization(data):
    """
    Creates direct visualization using the provided plot code.
    
    This visualization shows the relationship between max_distance and savings
    across different sets and cost models.
    """
    import matplotlib.pyplot as plt
    import hashlib
    import numpy as np
    import pandas as pd
    
    # Prepare data for plotting
    plot_data = data.copy()
    
    # Convert None values to string 'None' for clearer handling
    plot_data['formatted_distance'] = plot_data['max_distance'].apply(
        lambda x: 'None' if pd.isna(x) else x
    )
    
    # Create numeric version for plotting
    # Map 'None' to 0.05, and keep others as is
    plot_data['plot_x'] = plot_data['formatted_distance'].apply(
        lambda x: 0.05 if x == 'None' else x
    )
    
    plt.figure(figsize=(12, 8))
    
    # Plot points for each cost model
    for model in ['simple', 'complex']:
        model_data = plot_data[plot_data['cost_model'] == model]
        
        # Create scatter plot with different markers for each set_id
        for set_id in sorted(plot_data['set_id'].unique()):
            set_data = model_data[model_data['set_id'] == set_id]
            
            plt.scatter(set_data['plot_x'], 
                       set_data['savings_percentage'],
                       label=f'Set {set_id} ({model})' if set_id == 0 else "",
                       marker=['o', 's', '^', 'v', 'D', 'p', 'h', '*', '+', 'x'][set_id],
                       c='blue' if model == 'simple' else 'red',
                       alpha=0.6)

    # Customize the plot
    plt.xlabel('Max Distance')
    plt.ylabel('Savings Percentage (%)')
    plt.title('Savings vs Max Distance by Set and Cost Model')
    plt.grid(True, alpha=0.3)

    # Modify x-axis to show "None" instead of 0.05
    x_ticks = [0.02, 0.03, 0.04, 0.05]
    plt.xticks(x_ticks, ['0.02', '0.03', '0.04', 'NaN'])

    # Add legend with two entries for the models
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Simple Model', markersize=10),
              plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Complex Model', markersize=10)]
    plt.legend(handles=handles, loc='upper left')

    # Add text box explaining markers
    marker_text = "Markers by Set ID:\n" + \
                 "\n".join([f"Set {i}: {marker}" for i, marker in 
                           enumerate(['●', '■', '▲', '▼', '◆', 'p', 'h', '★', '+', '×'])])
    plt.text(1.15, 0.5, marker_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))

    # Debug info
    # print("\nData points by formatted distance:")
    # for dist in plot_data['formatted_distance'].unique():
    #     count = len(plot_data[plot_data['formatted_distance'] == dist])
    #     print(f"Distance {dist}: {count} data points")

    plt.tight_layout()

    # Save the plot with higher resolution
    plot_filename = f'package-products.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {plot_filename}")    


def create_column_description_file(csv_filename):
    """
    Creates a text file with column definitions for the CSV output.
    
    Args:
        csv_filename: Name of the CSV file to describe
    """
    txt_filename = csv_filename + '.txt'
    
    with open(txt_filename, 'w') as f:
        f.write(f"Column Definitions for {csv_filename}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write column definitions
        f.write("Column Name          | Description\n")
        f.write("-" * 60 + "\n")
        f.write("original_trades      | Number of individual option trades before grouping\n")
        f.write("grouped_trades       | Number of consolidated trades after grouping\n")
        f.write("original_cost        | Total transaction cost (USD) if executing all trades individually\n")
        f.write("grouped_cost         | Total transaction cost (USD) after grouping trades\n")
        f.write("savings              | Absolute cost reduction (USD) from grouping\n")
        f.write("savings_percentage   | Percentage cost reduction (%) from grouping\n")
        f.write("set_id               | Identifier for the option set (from input data)\n")
        f.write("max_distance         | Maximum allowed distance between strikes for grouping\n")
        f.write("                     | (None = no restriction on distance)\n")
        f.write("cost_model           | Transaction cost model used:\n")
        f.write("                     |   - simple: Tiered model (0.06% ≤$1M, 0.03% >$1M)\n")
        f.write("                     |   - complex: Logarithmic model ((11-0.5*ln(notional))/10000)\n")
        
        # Add additional metadata and explanation
        f.write("\n\nAnalysis Details:\n")
        f.write("-" * 20 + "\n")
        f.write("This analysis examines how grouping options at different strike prices\n")
        f.write("affects execution costs. The max_distance parameter controls risk by\n")
        f.write("limiting how far the grouped strike can be from original strikes.\n\n")
        
        f.write("Key Metrics:\n")
        f.write("- Trade reduction: Compare original_trades vs grouped_trades\n")
        f.write("- Cost efficiency: Examine savings_percentage across different parameters\n")
        f.write("- Risk vs reward: Higher max_distance allows more grouping but increases basis risk\n")
    
    print(f"Column descriptions saved to {txt_filename}")

    
#######################################################
# MAIN EXECUTION FLOW
#######################################################

def main():
    """
    Main execution flow implementing the complete option package optimization.
    
    Implementation of core requirements:
    1. "Read the attached data" - Loads and validates input
    2. "Provide groups for each dataset" - Processes each set
    3. "Evaluate several maximum distances" - Tests different constraints
    4. "Two alternatives for transaction costs" - Tests both models
    
    Analysis Framework:
    - Processes multiple sets independently
    - Evaluates different risk constraints
    - Compares cost models
    - Tracks grouping details
    - Generates execution instructions
    - Saves results for analysis
    """
    # Load and validate input data
    input_csv = 'products_to_package.csv'
    data = read_validate_csv(input_csv)
    
    # Setup test parameters from requirements
    unique_sets = sorted(data['set'].unique())
    max_distances = [0.02, 0.03, 0.04, None]  # From problem specification
    cost_models = {
        'simple': calculate_transaction_cost_simple,
        'complex': calculate_transaction_cost_complex
    }
    
    # Run optimization analysis
    results = []
    groupings = {}
    
    # TODO unpack nested loops
    # Open a file to store all trading instructions
    with open('trading_instructions.txt', 'w') as f:
        f.write("TRADING INSTRUCTIONS FOR ALL SCENARIOS\n")
        f.write("=====================================\n\n")
        
        idx = 0
        for set_id in unique_sets:
            set_data = data[data['set'] == set_id]
            
            for max_dist in max_distances:
                for model_name, cost_func in cost_models.items():
                    # Group and evaluate
                    grouped, group_details = group_options(set_data, max_dist, cost_func)
                    eval_results = evaluate_grouping(set_data, grouped, cost_func)
                    eval_results.update({
                        'set_id': set_id,
                        'max_distance': max_dist,
                        'cost_model': model_name
                    })
                    results.append(eval_results)
                    groupings[idx] = group_details
                    
                    # Capture trading instructions for this scenario
                    # Redirect stdout to capture the output of show_trading_instructions
                    original_stdout = sys.stdout
                    captured_output = io.StringIO()
                    sys.stdout = captured_output
                    
                    # Generate the trading instructions
                    show_trading_instructions(set_data, grouped)
                    
                    # Restore stdout
                    sys.stdout = original_stdout
                    instructions = captured_output.getvalue()
                    
                    # Write to file with scenario details
                    f.write(f"\n\nSCENARIO: Set {set_id}, Max Distance {max_dist if max_dist else 'None'}, Cost Model {model_name}\n")
                    f.write("".join(["-"] * 80) + "\n")
                    f.write(instructions)
                    f.write("\n" + "".join(["="] * 80) + "\n")
                    
                    idx += 1
    
    print(f"Trading instructions saved to trading_instructions.txt")
    
    # Create and analyze results
    results_df = pd.DataFrame(results)
    output = {
        'results': results_df,
        'groupings': groupings
    }
    
    output_csv = 'package-products.csv'
    results_df.to_csv(output_csv)
    print(f"Optimization analysis results saved to {output_csv}")
    
    output_pickle = 'package-products.pkl'
    results_df.to_pickle(output_pickle)
    print(f"Optimization analysis results saved to {output_pickle}")
    
    print(f"Optimization analysis results column descriptions saved to {output_pickle}")
    create_column_description_file(output_csv)  # Add this line

    # Scatter plot of results
    create_direct_visualization(results_df)

    # Debugging/sanity checks
    # Average savings percentage by max_distance and cost_model
    # print("\nAverage savings percentage by max_distance and cost_model:")
    # pivot = results_df.pivot_table(
    #     values='savings_percentage',
    #     index='max_distance',
    #     columns='cost_model',
    #     aggfunc='mean'
    # )
    # print(pivot)

    # # Show example grouping details
    # print("\nExample Grouping Details for first result:")
    # print(f"Result row 0:")
    # print(results_df.iloc[0])
    # print("\nCorresponding grouping details:")
    # print(groupings[0])

if __name__ == "__main__":
    main()