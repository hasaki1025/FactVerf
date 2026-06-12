import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gnn.dataset import FEVERGraphDataset
from gnn.preprocess_dataset import FEVERDataSet
from gnn.model import LLMSKAN
from utils.early_stopping import EarlyStopping
from utils.logger import get_logger

logger = get_logger()

def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    # For FEVER score, official evaluation relies on evidence retrieval matching.
    # Since this script trains the classification module, we use Accuracy to approximate
    # label accuracy assuming evidence is fixed. True FEVER score needs the official eval script.
    fever_score = acc 
    return {"acc": acc, "f1": f1, "fever_score": fever_score}

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch.y.cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc=desc):
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y)
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch.y.cpu().numpy())
        
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics

def main():
    parser = argparse.ArgumentParser(description="Train LLM-SKAN GNN Model")
    parser.add_argument("--config", type=str, default="config/gnn_config.yaml", help="Path to config file")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="FEVER", help="Name of the dataset (e.g. FEVER)")
    parser.add_argument("--dataset_dir", type=str, default="dataset/FEVER", help="Directory containing dataset files")
    
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save best model")
    args = parser.parse_args()

    # Construct paths based on dataset
    if args.dataset_name.upper() == "FEVER":
        train_raw_path = os.path.join(args.dataset_dir, "train.jsonl")
        train_claim_path = os.path.join(args.dataset_dir, "train_claim.jsonl")
        train_evidence_path = os.path.join(args.dataset_dir, "train_evidence.jsonl")
        
        val_raw_path = os.path.join(args.dataset_dir, "shared_task_dev.jsonl")
        val_claim_path = os.path.join(args.dataset_dir, "val_claim.jsonl")
        val_evidence_path = os.path.join(args.dataset_dir, "val_evidence.jsonl")
        
        # Test uses val for now
        test_raw_path = val_raw_path
        test_claim_path = val_claim_path
        test_evidence_path = val_evidence_path
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} is not fully supported yet. Please add its path parsing logic.")

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    train_cfg = config.get('train', {})
    
    batch_size = train_cfg.get('batch_size', 24)
    lr = train_cfg.get('learning_rate', 2e-4)
    weight_decay = train_cfg.get('weight_decay', 1e-5)
    epochs = train_cfg.get('epochs', 50)
    patience = train_cfg.get('patience', 5)
    log_dir = train_cfg.get('log_dir', 'runs/fever_gnn')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Datasets and Loaders
    logger.info("Loading Training Data...")
    train_raw = FEVERDataSet(train_raw_path, train_claim_path, train_evidence_path)
    train_ds = FEVERGraphDataset(train_raw)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    logger.info("Loading Validation Data...")
    val_raw = FEVERDataSet(val_raw_path, val_claim_path, val_evidence_path)
    val_ds = FEVERGraphDataset(val_raw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Model
    model = LLMSKAN(args.config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # TensorBoard and EarlyStopping
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=best_model_path)

    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, desc="Validating")
        
        # Logging
        logger.info(f"Train | Loss: {train_loss:.4f} | Acc: {train_metrics['acc']:.4f} | F1: {train_metrics['f1']:.4f}")
        logger.info(f"Val   | Loss: {val_loss:.4f} | Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f}")
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Acc/Train', train_metrics['acc'], epoch)
        writer.add_scalar('F1/Train', train_metrics['f1'], epoch)
        
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Acc/Val', val_metrics['acc'], epoch)
        writer.add_scalar('F1/Val', val_metrics['f1'], epoch)
        
        # Early Stopping checks validation loss
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered. Stopping training.")
            break

    writer.close()

    # Test Evaluation
    logger.info("Loading Test Data...")
    test_raw = FEVERDataSet(test_raw_path, test_claim_path, test_evidence_path)
    test_ds = FEVERGraphDataset(test_raw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    logger.info("Evaluating on Test Set with Best Model...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, desc="Testing")
    logger.info(f"Test  | Loss: {test_loss:.4f} | Acc: {test_metrics['acc']:.4f} | F1: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
