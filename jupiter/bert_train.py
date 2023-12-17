import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl
from sklearn.metrics import f1_score

class PretrainedBert(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler, dataloader_train, dataloader_validation, f1_score_func):
        super(PretrainedBert, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader_train = dataloader_train
        self.dataloader_validation = dataloader_validation
        self.f1_score_func = f1_score_func
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Log training loss
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Log validation loss
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True)

        # Debugging: Check the shapes
        print('Logits shape:', outputs.logits.shape)
        print('Labels shape:', labels.shape)

        # Additional metrics, e.g., F1 score
        predictions = outputs.logits.argmax(dim=-1).squeeze()
        print('Predictions shape:', predictions.shape)

        true_vals = labels

        # Additional check for dimensions
        if len(predictions.shape) != len(true_vals.shape):
            print("Mismatch in dimensions - predictions and true_vals")
            # Handle this discrepancy according to your model's requirements
        else:
            val_f1 = self.compute_f1_score(predictions.cpu().numpy(), true_vals.cpu().numpy())

            # Log F1 score
            self.log('val_f1', val_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def compute_f1_score(self, predictions, labels):
        f1 = f1_score(labels, predictions, average='weighted')
        return f1

# Assuming you have your model, optimizer, scheduler, and dataloaders set up

# Create Lightning model
lightning_model = PretrainedBert(model, optimizer, scheduler, dataloader_train, dataloader_validation, f1_score_func)

# Create PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=epochs,
    limit_train_batches=batch_size,
)
# Train the model
trainer.fit(lightning_model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_validation)
