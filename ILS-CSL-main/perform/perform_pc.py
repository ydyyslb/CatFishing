import numpy as np
import pandas as pd
import os
import sys
from scipy import stats
from typing import List, Dict, Any

# 为了确保正确导入，我们不使用sys.path.append
from hc.pc import pc_test
from hc.DAG import DAG

def perform_pc(d: str = None, 
             s: int = None, 
             r: int = None, 
             score: str = None,
             score_filepath: str = None,
             true_dag: np.ndarray = None,
             alpha: float = 0.01, 
             max_condition_set_size: int = 3,
             is_soft: bool = False,
             **kwargs) -> np.ndarray:
    """
    Perform enhanced PC algorithm test, following the same data loading approach as HC
    
    Args:
        d: dataset name
        s: sample size
        r: random seed/data index
        score: scoring method
        score_filepath: path to score file
        true_dag: true DAG adjacency matrix
        alpha: significance level for independence tests
        max_condition_set_size: maximum size of conditioning set
        is_soft: whether or not to use prior knowledge as soft constraints
        
    Returns:
        p x p adjacency matrix representing causal structure
    """
    if true_dag is None:
        raise ValueError("true_dag must be provided in config")
    
    # Load data following the same approach as HC
    try:
        # First try to load from CSV, which is the primary format
        csv_filename = f"data/csv/{d}_{s}_{r}.csv"
        print(f"Attempting to load data from CSV: {csv_filename}")
        
        if os.path.isfile(csv_filename):
            # Read data as in HC algorithm - use category type for Bayesian networks with discrete variables
            data_df = pd.read_csv(csv_filename, dtype='category')
            print(f"Successfully loaded CSV data with shape: {data_df.shape}")
            
            # Convert categorical data to numeric for PC algorithm
            for col in data_df.columns:
                # Convert each category to a numeric code
                data_df[col] = data_df[col].cat.codes
            
            # Convert to numpy array for PC algorithm
            data = data_df.values
            print(f"Converted categorical data to numeric, data shape: {data.shape}")
        else:
            # Fallback to txt format if csv not available
            txt_filename = f"data/{d}_{s}_{r}.txt"
            print(f"CSV not found, attempting to load from TXT: {txt_filename}")
            
            if os.path.isfile(txt_filename):
                try:
                    # Try loading assuming it's a numeric file
                    data = np.loadtxt(txt_filename)
                except ValueError:
                    # If that fails, try pandas which can better handle mixed types
                    data_df = pd.read_csv(txt_filename, header=None, delim_whitespace=True)
                    # Convert to numeric, errors='coerce' will convert non-numeric to NaN
                    data_df = data_df.apply(pd.to_numeric, errors='coerce')
                    # Fill NaN with 0 or other appropriate value
                    data_df = data_df.fillna(0)
                    data = data_df.values
                
                print(f"Successfully loaded TXT data with shape: {data.shape}")
            else:
                # Use score file as last resort
                if score_filepath and os.path.isfile(score_filepath):
                    print(f"Attempting to load from score file: {score_filepath}")
                    try:
                        # Try with numpy first
                        data = np.loadtxt(score_filepath)
                    except ValueError:
                        # If that fails, try with pandas
                        data_df = pd.read_csv(score_filepath, header=None)
                        data_df = data_df.apply(pd.to_numeric, errors='coerce')
                        data_df = data_df.fillna(0)
                        data = data_df.values
                    
                    print(f"Successfully loaded score data with shape: {data.shape}")
                else:
                    raise FileNotFoundError(f"Could not find data file for {d}_{s}_{r}")
        
        # Ensure the data has the right number of variables
        num_vars = true_dag.shape[0]
        if data.shape[1] != num_vars:
            print(f"Warning: Data variables ({data.shape[1]}) don't match DAG variables ({num_vars})")
            
            if data.shape[1] > num_vars:
                # Too many variables, take only what we need
                data = data[:, :num_vars]
            else:
                # Too few variables, this is a critical error
                raise ValueError(f"Data has fewer variables ({data.shape[1]}) than the DAG ({num_vars})")

        # Determine the optimal test method based on the dataset
        if d in ["asia", "cancer", "child", "alarm", "mildew", "barley", "water", "insurance"]:
            # These are known to be categorical datasets
            test_method = 'g2'
        else:
            # For other datasets, let the algorithm auto-detect
            test_method = 'auto'
        
        # Optimize alpha based on dataset size
        if s <= 500:
            # For smaller datasets, use less strict threshold
            alpha = 0.05
        elif s <= 2000:
            # Medium datasets
            alpha = 0.01
        else:
            # Large datasets can afford to be more stringent
            alpha = 0.005
            
        # Optimize condition set size based on sample size
        if s < 1000:
            # For smaller sample sizes, limit the condition set to avoid sparse data
            max_condition_set_size = min(2, max_condition_set_size) 
        elif s < 5000:
            max_condition_set_size = min(3, max_condition_set_size)
                
        # Run PC algorithm with the loaded data and optimized parameters
        print(f"Running PC algorithm with alpha={alpha}, max_condition_set_size={max_condition_set_size}, test_method={test_method}")
        
        # Handle any existing edges or forbidden edges from prior iterations
        exist_edges = kwargs.get('exist_edges', [])
        forb_edges = kwargs.get('forb_edges', [])
        
        # Print prior knowledge if available
        if exist_edges:
            print(f"Using {len(exist_edges)} existing edges as prior knowledge")
        if forb_edges:
            print(f"Using {len(forb_edges)} forbidden edges as prior knowledge")
        
        # Run the enhanced PC algorithm
        result = pc_test(data, alpha=alpha, max_condition_set_size=max_condition_set_size, 
                         stable=True, test_method=test_method)
        
        # Apply prior knowledge constraints if specified
        if exist_edges and (is_soft or len(exist_edges) > 0):
            for i, j in exist_edges:
                # Add confirmed edges to result
                result[i, j] = 1
                # Remove any reverse edges to maintain DAG property
                result[j, i] = 0
                
        if forb_edges and (is_soft or len(forb_edges) > 0):
            for i, j in forb_edges:
                # Remove forbidden edges
                result[i, j] = 0
                
        print(f"PC algorithm result shape: {result.shape}")
        return result
        
    except Exception as e:
        print(f"Error loading data or running PC algorithm: {e}")
        # Create empty DAG as fallback
        print("Creating empty DAG as fallback")
        result = np.zeros_like(true_dag)
        return result 