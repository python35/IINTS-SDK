from iints.research.glucose_model import build_model, get_loss_fn
from iints.research.dataset import GlucoseForecastingDataset
from iints.research.forecasting import TimeSeriesPredictor
import torch
import yaml

def train():
    print("Loading config...")
    with open("models/iints-glucose-forecast-v0/dataset/glucose_model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Overrides for speed and safety
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 512
    
    print("Loading dataset...")
    dataset = GlucoseForecastingDataset(
        data_path="models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.parquet",
        history_minutes=config["predictor"]["history_minutes"],
        horizon_minutes=config["predictor"]["horizon_minutes"],
        time_step_minutes=config["predictor"]["time_step_minutes"],
        feature_columns=config["predictor"]["feature_columns"],
        target_column=config["predictor"]["target_column"],
        split="train"
    )
    
    print("Building model...")
    # Manually instantiate to avoid cli wrappers
    input_size = len(config["predictor"]["feature_columns"])
    model = build_model(config, input_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = get_loss_fn(config)
    
    print("Training 1 fast epoch (CPU)...")
    # Grab a few batches manually instead of DataLoader
    model.train()
    total_loss = 0.0
    
    # Only train on a subset to prevent timeout
    indices = torch.randperm(len(dataset))[:5120] 
    
    for i in range(0, len(indices), 512):
        batch_idx = indices[i:i+512]
        # Gather batch
        x_list, y_list = [], []
        for idx in batch_idx:
            x, y = dataset[idx.item()]
            x_list.append(x)
            y_list.append(y)
            
        x_batch = torch.stack(x_list)
        y_batch = torch.stack(y_list)
        
        optimizer.zero_grad()
        preds = model(x_batch)
        
        # PINN Loss requires x_batch context
        loss = loss_fn(preds, y_batch, x_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Batch {i//512} Loss: {loss.item():.4f}")
        
    print(f"Final PINN Loss: {total_loss:.4f}")
    
    print("Saving predictor...")
    predictor = TimeSeriesPredictor(model, config)
    predictor.save("models/iints-glucose-forecast-v0/predictor.pt")
    print("Done!")

if __name__ == "__main__":
    train()
