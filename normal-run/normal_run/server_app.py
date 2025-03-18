"""normal-run: A Flower / PyTorch app."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures only

import time
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
import torch.nn.functional as F
from normal_run.task import LeNet, get_weights, init_weights
import numpy as np
import os

RESULTS_DIRECTORY = "experiment_results"
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def simulate_dlg_attack(model, original_params, original_gradient, n_iter=300):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model = model.to(device)
    
    dummy_data = torch.randn(1, 3, 32, 32).to(device).requires_grad_(True)
    dummy_label = torch.randn(1, 50).to(device).requires_grad_(True)  # 10 classes for CIFAR-10
    
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    
    history = []
    
    for i in range(n_iter):
        def closure():
            optimizer.zero_grad()
        
            dummy_pred = model(dummy_data) 
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot_label) 
            dummy_gradient = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            
            grad_diff = 0
            for gx, gy in zip(dummy_gradient, original_gradient): 
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()    
            return grad_diff
            
        optimizer.step(closure)
        if i % 10 == 0: 
            img_tensor = dummy_data[0].detach().cpu().clamp(0, 1)
            transform = transforms.ToPILImage()
            img = transform(img_tensor)
            history.append(img)
    return dummy_data, history

class FedAvgNormal(FedAvg):
    def __init__(self, *args, num_rounds=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics = {}
        self.start_time = time.time()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LeNet().to(self.device)
        self.num_rounds = num_rounds
    def aggregate_fit(self, server_round, results, failures):
        round_start_time = time.time()
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_result is None:
            return None
        
        params, metrics = aggregated_result
        
        
        if results and len(results)>0:
            client_idx = np.random.randint(0, len(results))
            client_params = parameters_to_ndarrays(results[client_idx][1].parameters)

            
            params_dict = zip(self.model.state_dict().keys(), client_params)
            state_dict = {k: torch.tensor(v) for k,v in params_dict}
            self.model.load_state_dict(state_dict)
            
            sample_data = torch.randn(1, 3, 32, 32).to(self.device)
            sample_label = torch.randint(0, 50, (1,)).to(self.device)
            sample_onehot = F.one_hot(sample_label, num_classes=50).float()
            
            pred = self.model(sample_data)
            loss = cross_entropy_for_onehot(pred, sample_onehot)
            original_grads = torch.autograd.grad(loss, self.model.parameters())
            original_grads = [g.detach() for g in original_grads]
            
            print(f"\nSimulating DLG attack for round {server_round}")
            _, history = simulate_dlg_attack(self.model, client_params, original_grads)
            
            if len(history) > 0:
                plt.figure(figsize=(12, 4))
                for i in range(min(6, len(history))):
                    plt.subplot(1, 6, i + 1)
                    plt.imshow(history[i])
                    plt.title(f"Iter {i*20}")
                    plt.axis('off')
                plt.savefig(f"{RESULTS_DIRECTORY}/simulated_dlg_at_round_{server_round}.png")
                plt.close()
            
            round_time = time.time() - round_start_time
            total_time = time.time() - self.start_time
            
            self.round_metrics[server_round] = {
                "round_time": round_time,
                "total_time": total_time,
                **metrics
            }
            
            print(f"Round {server_round} completed in {round_time:.2f}s\n Total time elapsed: {total_time:.2f}s")
            print(f"Metrics: {metrics}")
            
        return params, metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        result = super().aggregate_evaluate(server_round, results, failures)
        
        if result is not None:
            loss, metrics = result

            if metrics:
                if server_round in self.round_metrics:
                    self.round_metrics[server_round].update(metrics)
                else:
                    self.round_metrics[server_round] = metrics
                    
            print(f" Round {server_round} Eval:\n - Loss: {loss:.4f}\n Metrics: {metrics}") 

        if server_round == self.num_rounds:
            self._save_final_metrics()
        
        return result
    
    def _save_final_metrics(self):
        import csv
        with open(f"{RESULTS_DIRECTORY}/metrics.csv", 'w',newline='') as csv_file:
            field_names = ['round', 'time', 'loss', 'accuracy']
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()
            
            for round_num, metrics in self.round_metrics.items():
                writer.writerow({
                    'round' : round_num,
                    'time' : metrics.get('total_time', 0),
                    'loss' : metrics.get('loss', 0),
                    'accuracy' : metrics.get('accuracy', 0) 
                })
        
        rounds = list(self.round_metrics.keys())
        accuracy_list = [metrics.get('accuracy', 0) for metrics in self.round_metrics.values()]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, accuracy_list, marker='o')
        plt.title('Model Accuracy by Round')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(f"{RESULTS_DIRECTORY}/accuracy_plot.png")
        plt.close()
        
        print(f"\nExperiment completed! Results saved to {RESULTS_DIRECTORY}/") 
            
def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    model = LeNet()
    model = init_weights(model)
    parameters = ndarrays_to_parameters([np.array(v.cpu()) for k, v in model.state_dict().items()])
    
    # Define strategy
    strategy = FedAvgNormal(
        fraction_fit = fraction_fit,
        fraction_evaluate = 1.0,
        min_available_clients = 2,
        initial_parameters = parameters,
        num_rounds = num_rounds
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
