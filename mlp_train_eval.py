import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from models.mlp_model import MLPTrainer
from dataloaders.mlp_dataloader import prepare_data

def main():
    input_dim = 320
    hidden_dim = 256
    output_dim = 10
    dropout_rate = 0.3
    batch_size = 32
    max_epochs = 100
    n_splits = 5 

    # Load data
    embeddings = torch.load('data/embeds/protein_representations.pt')  # Relative path
    labels = torch.tensor(np.load('data/embeds/labels_cath.npy')).long()  # Relative path

    # Initialize K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Track validation scores
    validation_scores = []

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(embeddings)):
        print(f"Fold {fold+1}/{n_splits}")

        # Prepare data loaders for this fold
        train_loader, val_loader = prepare_data(embeddings, labels, train_idx, val_idx, batch_size=batch_size)

        # Initialize model
        model = MLPTrainer(input_dim, hidden_dim, output_dim, dropout_rate)

        # Initialize CSVLogger
        csv_logger = CSVLogger("logs", name=f"my_mlp_model_fold_{fold+1}")

        # Set up checkpointing to save the best model
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f'checkpoints/mlp/fold_{fold+1}',
            filename='best-checkpoint',
            save_top_k=1,
            mode='min'
        )

        # Set up learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # Initialize trainer with the logger
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, lr_monitor],
            logger=csv_logger,
            enable_progress_bar=True
        )

        # Train the model
        trainer.fit(model, train_loader, val_loader)

        # Load the best model checkpoint and validate
        best_model_path = checkpoint_callback.best_model_path
        best_model = MLPTrainer.load_from_checkpoint(best_model_path, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout_rate=dropout_rate)
        
        # Evaluate on validation set
        val_loss = trainer.validate(best_model, val_loader)[0]['val_loss']
        validation_scores.append(val_loss)

    # Print the cross-validation results
    avg_val_loss = np.mean(validation_scores)
    print(f"Average Validation Loss across {n_splits} folds: {avg_val_loss:.4f}")

if __name__ == "__main__":
    main()
