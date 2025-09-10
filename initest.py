#!/usr/bin/env python3
"""
Monte Carlo simulation for the frog's random walk problem.

This script implements the solution to the frog lily pad problem as specified
in the problem statement.
"""

import random
import numpy as np


def simulate_frog_walk():
    """
    Simulate a single frog walk until it reaches lily pad 50 for the first time.
    
    Returns:
        bool: True if the frog visited all lily pads except 50 before reaching 50
    """
    current_position = 0
    visited_pads = {0}  # Start at lily pad 0
    
    while current_position != 50:
        # Choose direction: -1 (left) or +1 (right) with equal probability
        direction = random.choice([-1, 1])
        
        # Move to the next lily pad (circular arrangement)
        current_position = (current_position + direction) % 100
        
        # Add current position to visited pads
        visited_pads.add(current_position)
    
    # Check if all lily pads except 50 have been visited
    # There are 100 lily pads total (0-99), so we need 100 visited pads (including 50)
    return len(visited_pads) == 100


def monte_carlo_simulation(num_trials=1000000):
    """
    Run Monte Carlo simulation for the frog's random walk problem.
    
    Args:
        num_trials (int): Number of simulation trials to run
        
    Returns:
        float: Estimated probability
    """
    successful_trials = 0
    
    print(f"Running Monte Carlo simulation with {num_trials:,} trials...")
    
    # Run simulation trials
    for trial in range(num_trials):
        if trial % 100000 == 0 and trial > 0:
            current_prob = successful_trials / trial
            print(f"Progress: {trial:,} trials completed. Current estimate: {current_prob:.6f}")
        
        if simulate_frog_walk():
            successful_trials += 1
    
    probability = successful_trials / num_trials
    return probability


if __name__ == "__main__":
    print("Frog's Random Walk Monte Carlo Simulation")
    print("=" * 50)
    print()
    print("Problem: Find the probability that when a frog lands on lily pad 50")
    print("for the first time, it has visited every other lily pad (0-49, 51-99).")
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run full simulation with 1,000,000 trials as specified in the problem
    probability = monte_carlo_simulation(1000000)
    
    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Number of trials: 1,000,000")
    print(f"Estimated probability: {probability:.6f}")
    print(f"Estimated probability: {probability:.4%}")
    
    # Calculate confidence interval
    confidence_interval = 1.96 * np.sqrt(probability * (1 - probability) / 1000000)
    print(f"95% Confidence interval: [{probability - confidence_interval:.6f}, {probability + confidence_interval:.6f}]")
