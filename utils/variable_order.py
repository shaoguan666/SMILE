import torch
from data.feature_registry import get_candidate_systems, get_feature_names

def get_variable_order(dataset_name):
    """
    Get the physical permutation order of variables for 2D CNN (cross-variable) 
    in MissingPatternEncoder to group physiologically related features.
    
    Returns:
        var_order_idx: (V,) tensor for pre-CNN permutation
        inv_order_idx: (V,) tensor for post-CNN inverse permutation
    """
    
    systems = get_candidate_systems(dataset_name)
    features = get_feature_names(dataset_name)
        
    num_vars = len(features)
    
    # 1. Collect ordered indices based on predefined physiological systems
    ordered_indices = []
    seen = set()
    for sys_name, idx_list in systems.items():
        for idx in idx_list:
            if idx < num_vars and idx not in seen:
                ordered_indices.append(idx)
                seen.add(idx)
                
    # 2. Append unassigned isolated variables
    for idx in range(num_vars):
        if idx not in seen:
            ordered_indices.append(idx)
            seen.add(idx)
            
    assert len(ordered_indices) == num_vars, "Mismatch in number of variables."
            
    var_order_idx = torch.tensor(ordered_indices, dtype=torch.long)
    inv_order_idx = torch.argsort(var_order_idx)
    
    return var_order_idx, inv_order_idx
