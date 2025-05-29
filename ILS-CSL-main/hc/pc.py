import numpy as np
from scipy import stats
from typing import List, Set, Tuple, Dict, Optional, Union
import networkx as nx
from .accessory import *
import itertools
from collections import defaultdict, deque

class PC:
    def __init__(self, data: np.ndarray, alpha: float = 0.01, max_condition_set_size: int = None, 
                 stable: bool = True, test_method: str = 'auto'):
        """
        Initialize PC algorithm
        
        Args:
            data: n x p matrix where n is number of samples and p is number of variables
            alpha: significance level for independence tests
            max_condition_set_size: maximum size of conditioning set (None means no limit)
            stable: whether to use the stable version of PC (PC-Stable)
            test_method: 'g2' for G^2 test (categorical), 'pearson' for Pearson correlation test (continuous),
                         'auto' to automatically choose based on data
        """
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.alpha = alpha
        self.max_condition_set_size = max_condition_set_size if max_condition_set_size is not None else self.n_vars - 2
        self.stable = stable
        
        # Determine data type and test method
        if test_method == 'auto':
            # Check if data appears to be categorical (few unique values)
            is_categorical = True
            for col in range(self.n_vars):
                unique_vals = np.unique(self.data[:, col])
                if len(unique_vals) > 10 or (len(unique_vals) > 0.1 * self.n_samples):
                    is_categorical = False
                    break
                    
            self.test_method = 'g2' if is_categorical else 'pearson'
            print(f"Auto-selected test method: {self.test_method}")
        else:
            self.test_method = test_method
        
        # Initialize fully connected undirected graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.n_vars))
        for i in range(self.n_vars):
            for j in range(i+1, self.n_vars):
                self.graph.add_edge(i, j)
                
        # Store separating sets
        self.separating_sets = {}
        
    def _g2_ci_test(self, x: int, y: int, condition_set: Set[int]) -> Tuple[bool, float]:
        """
        Perform conditional independence test using G^2 test (for categorical data)
        
        Args:
            x: index of first variable
            y: index of second variable
            condition_set: set of conditioning variables
            
        Returns:
            (is_independent, p_value)
        """
        # Extract the variables we need
        variables = list(condition_set) + [x, y]
        selected_data = self.data[:, variables]
        
        # Count unique values for each variable to determine dimensions
        dims = [len(np.unique(self.data[:, v])) for v in variables]
        
        # Handle the case where a variable has only one value (no information)
        if any(d <= 1 for d in dims):
            return True, 1.0
            
        # Generate contingency tables and calculate G^2 statistic
        if len(condition_set) == 0:
            # Simple case: no conditioning
            contingency = np.zeros((dims[-2], dims[-1]))
            
            # Fill contingency table
            for i in range(len(selected_data)):
                x_val = int(selected_data[i, 0])
                y_val = int(selected_data[i, 1])
                contingency[x_val, y_val] += 1
            
            # Calculate chi-square and p-value (G^2 has same asymptotic distribution)
            _, p_value, _, _ = stats.chi2_contingency(contingency)
            
        else:
            # Complex case: with conditioning variables
            # We compute G^2 by summing across all configurations of conditioning variables
            g2 = 0
            df = (dims[-2] - 1) * (dims[-1] - 1)  # degrees of freedom
            n_configs = np.prod(dims[:-2])
            
            # If there are too many configurations (sparse table), use approximation
            if n_configs > self.n_samples / 5:
                # Use a more efficient approach for sparse tables
                # Group by conditioning variables
                cond_indices = tuple(range(len(condition_set)))
                xy_indices = (len(condition_set), len(condition_set) + 1)
                
                # Accumulate G^2 for each conditioning configuration
                counts = defaultdict(lambda: np.zeros((dims[-2], dims[-1])))
                
                for row in selected_data:
                    cond_values = tuple(int(row[i]) for i in cond_indices)
                    x_val = int(row[xy_indices[0]])
                    y_val = int(row[xy_indices[1]])
                    counts[cond_values][x_val, y_val] += 1
                
                # Calculate G^2 with continuity correction for sparse tables
                g2 = 0
                for table in counts.values():
                    if np.sum(table) >= 5:  # Only include tables with sufficient data
                        _, p, _, _ = stats.chi2_contingency(table + 0.5)  # Add 0.5 for continuity correction
                        g2 += p
                
                # Average p-value across tables
                p_value = g2 / max(1, len(counts))
                
            else:
                # Standard approach for well-populated tables
                # Build multi-dimensional contingency table
                contingency = np.zeros(dims)
                
                # Fill contingency table
                for row in selected_data:
                    indices = tuple(int(val) for val in row)
                    index = np.ravel_multi_index(indices, dims)
                    contingency.flat[index] += 1
                
                # Calculate G^2 statistic
                # Compute expected frequencies and G^2
                marginal_xy = np.sum(contingency, axis=tuple(range(len(condition_set))))
                
                # Check if marginals are zero
                if np.any(np.sum(marginal_xy, axis=0) == 0) or np.any(np.sum(marginal_xy, axis=1) == 0):
                    return True, 1.0
                
                # Calculate p-value using chi-square distribution
                _, p_value, _, _ = stats.chi2_contingency(marginal_xy)
            
        return p_value > self.alpha, p_value
        
    def _pearson_ci_test(self, x: int, y: int, condition_set: Set[int]) -> Tuple[bool, float]:
        """
        Perform conditional independence test using partial correlation
        
        Args:
            x: index of first variable
            y: index of second variable
            condition_set: set of conditioning variables
            
        Returns:
            (is_independent, p_value)
        """
        if len(condition_set) == 0:
            # Simple correlation test
            corr, p_value = stats.pearsonr(self.data[:, x], self.data[:, y])
        else:
            # Partial correlation test
            condition_vars = list(condition_set)
            x_resid = self._get_residuals(x, condition_vars)
            y_resid = self._get_residuals(y, condition_vars)
            corr, p_value = stats.pearsonr(x_resid, y_resid)
            
        return p_value > self.alpha, p_value
    
    def _ci_test(self, x: int, y: int, condition_set: Set[int]) -> Tuple[bool, float]:
        """Perform appropriate conditional independence test based on data type"""
        if self.test_method == 'g2':
            return self._g2_ci_test(x, y, condition_set)
        else:  # 'pearson'
            return self._pearson_ci_test(x, y, condition_set)
    
    def _get_residuals(self, target: int, condition_vars: List[int]) -> np.ndarray:
        """Get residuals after regressing target on condition variables"""
        X = self.data[:, condition_vars]
        y = self.data[:, target]
        
        # Add constant term
        X = np.column_stack([np.ones(len(X)), X])
        
        # Solve least squares
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Calculate residuals
        residuals = y - X @ beta
        return residuals
    
    def _find_separating_sets(self):
        """Find separating sets for each pair of variables using PC-Stable approach"""
        # Get all adjacent node pairs
        adj_pairs = list(self.graph.edges())
        
        # Start with empty conditioning set
        condition_size = 0
        
        while condition_size <= self.max_condition_set_size and adj_pairs:
            # PC-Stable: work with the adjacency structure at the beginning of each phase
            if self.stable:
                # Make a copy of current graph structure
                current_neighbors = {}
                for node in range(self.n_vars):
                    current_neighbors[node] = set(self.graph.neighbors(node))
            
            # Group edges to remove by the first node
            edges_to_check = defaultdict(list)
            for x, y in adj_pairs:
                edges_to_check[x].append(y)
                edges_to_check[y].append(x)
            
            removed_edges = []  # Keep track of edges to remove
            
            # For each node
            for node in range(self.n_vars):
                # Get neighbors in current iteration
                if self.stable:
                    neighbors = current_neighbors[node]
                else:
                    neighbors = set(self.graph.neighbors(node))
                
                # For each pair of adjacent nodes
                for neighbor in edges_to_check[node]:
                    # Skip if edge was already marked for removal
                    if (node, neighbor) in removed_edges or (neighbor, node) in removed_edges:
                        continue
                        
                    # Skip if edge is already removed
                    if not self.graph.has_edge(node, neighbor):
                        continue
                    
                    # Get all possible conditioning sets
                    if self.stable:
                        potential_separators = current_neighbors[node] - {neighbor}
                    else:
                        potential_separators = set(self.graph.neighbors(node)) - {neighbor}
                    
                    found_separator = False
                    
                    # Check all conditioning sets of current size
                    for condition_set in self._get_condition_sets(potential_separators, condition_size):
                        is_independent, p_value = self._ci_test(node, neighbor, condition_set)
                        
                        if is_independent:
                            # Mark edge for removal
                            removed_edges.append((node, neighbor))
                            
                            # Store separating set
                            self.separating_sets[(node, neighbor)] = condition_set
                            self.separating_sets[(neighbor, node)] = condition_set
                            
                            found_separator = True
                            break
                    
                    # No need to check other conditioning sets if we found one
                    if found_separator:
                        continue
            
            # Remove edges identified in this phase
            for x, y in removed_edges:
                if self.graph.has_edge(x, y):  # Check again just to be safe
                    self.graph.remove_edge(x, y)
            
            # Update adjacency pairs for next iteration
            adj_pairs = list(self.graph.edges())
            
            # Increment conditioning set size
            condition_size += 1
            
    def _get_condition_sets(self, variables: Set[int], size: int) -> List[Set[int]]:
        """Get all possible conditioning sets of given size"""
        if size == 0:
            return [set()]
        if size > len(variables):
            return []
            
        return [set(comb) for comb in itertools.combinations(variables, size)]
    
    def _is_clique(self, nodes: Set[int]) -> bool:
        """Check if the nodes form a clique (fully connected subgraph)"""
        for i, j in itertools.combinations(nodes, 2):
            if not self.graph.has_edge(i, j):
                return False
        return True
    
    def _orient_edges(self):
        """Orient edges using rules from PC algorithm with enhanced Meek rules"""
        try:
            # Convert to directed graph
            self.dag = nx.DiGraph()
            self.dag.add_nodes_from(range(self.n_vars))
            
            # Create an undirected copy of the graph for reference
            self.undirected = nx.Graph(self.graph)
            
            # Start with unoriented edges (create a CPDAG)
            for x, y in self.graph.edges():
                self.dag.add_edge(x, y)
                self.dag.add_edge(y, x)
                
            # Rule 1: Orient v-structures (colliders)
            self._orient_v_structures()
            
            # Apply Meek rules until convergence
            repeat = True
            while repeat:
                repeat = False
                # Rule 2: Orient based on acyclicity
                repeat |= self._meek_rule_1()
                
                # Rule 3: Orient based on avoiding new v-structures
                repeat |= self._meek_rule_2()
                
                # Rule 4: Orient based on avoiding cycles with a certain pattern
                repeat |= self._meek_rule_3()
                
        except Exception as e:
            print(f"Error in orient_edges: {e}")
            # Create empty graph in case of error
            self.dag = nx.DiGraph()
            self.dag.add_nodes_from(range(self.n_vars))
    
    def _orient_v_structures(self) -> bool:
        """Orient v-structures (colliders)"""
        oriented = False
        
        # Check for all possible v-structures (x -> y <- z)
        for y in range(self.n_vars):
            # Consider all nodes adjacent to y
            neighbors = list(self.undirected.neighbors(y))
            
            # For each pair of neighbors x and z of y
            for x, z in itertools.combinations(neighbors, 2):
                # If x and z are not adjacent
                if not self.undirected.has_edge(x, z):
                    # Check if y is not in the separating set of x and z
                    if (x, z) in self.separating_sets and y not in self.separating_sets[(x, z)]:
                        # Orient as v-structure: x -> y <- z
                        if self.dag.has_edge(y, x):
                            self.dag.remove_edge(y, x)
                            oriented = True
                        if self.dag.has_edge(y, z):
                            self.dag.remove_edge(y, z)
                            oriented = True
        
        return oriented
    
    def _meek_rule_1(self) -> bool:
        """
        Meek Rule 1: If i -> j - k, and i and k are not adjacent, then orient j -> k.
        This avoids creating new v-structures.
        """
        oriented = False
        
        for j in range(self.n_vars):
            # Find pairs of neighbors where one edge is directed toward j and the other is undirected
            incoming = [i for i in range(self.n_vars) if self.dag.has_edge(i, j) and not self.dag.has_edge(j, i)]
            undirected = [k for k in range(self.n_vars) if self.dag.has_edge(j, k) and self.dag.has_edge(k, j)]
            
            for i in incoming:
                for k in undirected:
                    # If i and k are not adjacent
                    if not self.dag.has_edge(i, k) and not self.dag.has_edge(k, i):
                        # Orient j -> k
                        self.dag.remove_edge(k, j)
                        oriented = True
        
        return oriented
    
    def _meek_rule_2(self) -> bool:
        """
        Meek Rule 2: If i -> j -> k and i - k, then orient i -> k.
        This avoids creating cycles.
        """
        oriented = False
        
        for j in range(self.n_vars):
            # Find directed edges i -> j and j -> k
            incoming = [i for i in range(self.n_vars) if self.dag.has_edge(i, j) and not self.dag.has_edge(j, i)]
            outgoing = [k for k in range(self.n_vars) if self.dag.has_edge(j, k) and not self.dag.has_edge(k, j)]
            
            for i in incoming:
                for k in outgoing:
                    # If i and k have an undirected edge
                    if self.dag.has_edge(i, k) and self.dag.has_edge(k, i):
                        # Orient i -> k
                        self.dag.remove_edge(k, i)
                        oriented = True
        
        return oriented
    
    def _meek_rule_3(self) -> bool:
        """
        Meek Rule 3: If i - j and i - k and j -> l and k -> l and i and l are not adjacent,
        then orient i -> j and i -> k.
        This avoids creating a new v-structure.
        """
        oriented = False
        
        for i in range(self.n_vars):
            # Find pairs of neighbors with undirected edges
            undirected = [j for j in range(self.n_vars) if self.dag.has_edge(i, j) and self.dag.has_edge(j, i)]
            
            # Check all pairs of those neighbors
            for j, k in itertools.combinations(undirected, 2):
                # Find common children of j and k
                j_children = [l for l in range(self.n_vars) if self.dag.has_edge(j, l) and not self.dag.has_edge(l, j)]
                k_children = [l for l in range(self.n_vars) if self.dag.has_edge(k, l) and not self.dag.has_edge(l, k)]
                
                common_children = set(j_children) & set(k_children)
                
                for l in common_children:
                    # If i and l are not adjacent
                    if not self.dag.has_edge(i, l) and not self.dag.has_edge(l, i):
                        # Orient i -> j and i -> k
                        self.dag.remove_edge(j, i)
                        self.dag.remove_edge(k, i)
                        oriented = True
        
        return oriented
    
    def _has_new_vstructure(self) -> bool:
        """Check if graph has any v-structures (deprecated, using Meek rules instead)"""
        try:
            for x in self.dag.nodes():
                successors = list(self.dag.successors(x))
                for y in successors:
                    for z in successors:
                        if y != z and not (self.dag.has_edge(y, z) or self.dag.has_edge(z, y)):
                            return True
            return False
        except nx.NetworkXError as e:
            print(f"Error checking v-structures: {e}")
            return False
    
    def fit(self) -> nx.DiGraph:
        """
        Run PC algorithm to learn causal structure
        
        Returns:
            Directed acyclic graph representing causal structure
        """
        print(f"Starting PC algorithm (stable={self.stable}, test={self.test_method}, alpha={self.alpha})")
        
        # Step 1: Find separating sets
        self._find_separating_sets()
        
        print(f"Skeleton discovery complete: found {len(self.graph.edges())} edges")
        
        # Step 2: Orient edges
        self._orient_edges()
        
        print(f"Edge orientation complete: found {sum(1 for _ in self.dag.edges() if not self.dag.has_edge(list(self.dag.predecessors(_))[0], _))} directed edges")
        
        return self.dag
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Convert DAG to adjacency matrix"""
        adj_matrix = np.zeros((self.n_vars, self.n_vars))
        for x, y in self.dag.edges():
            # Only include edges that are definitely directed (not part of undirected component)
            if not self.dag.has_edge(y, x):
                adj_matrix[x, y] = 1
        return adj_matrix

def pc_test(data: np.ndarray, alpha: float = 0.01, max_condition_set_size: int = None, 
           stable: bool = True, test_method: str = 'auto') -> np.ndarray:
    """
    Run enhanced PC algorithm on data
    
    Args:
        data: n x p matrix where n is number of samples and p is number of variables
        alpha: significance level for independence tests
        max_condition_set_size: maximum size of conditioning set
        stable: whether to use the stable version of PC (PC-Stable)
        test_method: 'g2' for G^2 test (categorical), 'pearson' for Pearson correlation test (continuous),
                     'auto' to automatically choose based on data
        
    Returns:
        p x p adjacency matrix representing causal structure
    """
    try:
        print(f"Starting enhanced PC algorithm with alpha={alpha}, max_condition_set_size={max_condition_set_size}, stable={stable}")
        print(f"Data shape: {data.shape}")
        
        # Initialize PC algorithm
        pc = PC(data, alpha, max_condition_set_size, stable, test_method)
        
        # Run algorithm
        pc.fit()
        
        # Get result
        result = pc.get_adjacency_matrix()
        print(f"PC algorithm completed successfully, result shape: {result.shape}")
        return result
    except Exception as e:
        print(f"Error in PC algorithm: {e}")
        # Return empty graph in case of error
        n_vars = data.shape[1]
        return np.zeros((n_vars, n_vars)) 