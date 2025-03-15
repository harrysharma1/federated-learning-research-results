import flwr as fl 
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import(
    FitRes,
    Parameters,
    Scalar,
    FitIns,
    EvaluateIns,
    EvaluateRes,
    NDArrays,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
import numpy as np

class SecureAggregationProtocol(FedAvg):
    """_summary_

    Args:
        FedAvg (_type_): _description_
    """
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def aggregate(self, server_round:int, results: List[Tuple[fl.server.client_manager.ClientProxy, FitRes]], failures: List[Union[Tuple[fl.server.client_manager.ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results
        ]
        
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in weights_results
        ]
        
        total_examples = sum([num_examples for _, num_examples in weights_results])
        
        aggregated_weights = [
            np.sum(
                [weighted_layer[i] for weighted_layer in weighted_weights], axis=0
            )/ total_examples
            for i in range(len(weighted_weights[0]))
        ]
        
        return ndarrays_to_parameters(aggregated_weights), {}