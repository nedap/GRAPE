import wandb
import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np

from typing import Any
from torchvision import transforms
from sklearn.metrics import classification_report

from datasets.data_transforms import PointcloudScaleAndTranslate
from tools.builder import *
from utils.misc import modified_coefficient_of_variation, to_cpu

train_transforms = transforms.Compose(
    [
        PointcloudScaleAndTranslate(),
    ]
)


class MAELightningModule(pl.LightningModule):
    """
    Lightning Module for Point-MAE.
    Masked Autoencoder for pointcloud Self Supervised Learning. 
    """

    def __init__(self, cfg, args) -> None:
        super(MAELightningModule, self).__init__()
        self.base_model = model_builder(cfg.model)
        self.cfg = cfg
        self.wandb = args.wandb
        self.unmasked_logged = False

    def forward(self, points, vis=False):
        return self.base_model(points, vis=vis)

    def training_step(self, batch, batch_idx):
        points = train_transforms(batch)
        loss = self.base_model(points)

        if self.wandb is not None:
            # Log current learning rate
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('Learning Rate', current_lr)

            # Calculate and log the L2 norm of model parameters
            param_norm = torch.norm(torch.stack([torch.norm(p) for p in self.parameters()]), 2)
            self.log('L2 Norm of the Model parameters', param_norm)

            # Calculate and log the L2 norm of gradients
            gradients = [p.grad for p in self.parameters() if p.grad is not None]
            if gradients:
                grad_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients]), 2)
                self.log('L2 Norm of the Gradients', grad_norm)

            # Log training loss
            self.log('Training Loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.wandb:
            # if vis=True, model returns pcd data next to the loss
            output = self.base_model(batch, vis=True)

            # Move all tensors to CPU
            (reconstructed_pcd, unmasked_points, centers,
             predicted_points, points, masked, latent, loss) = to_cpu(*output)

            # Log point clouds
            self.log_pointclouds(reconstructed_pcd, unmasked_points, centers,
                                 predicted_points, points, masked, latent)

            # Log coefficient of variation
            self.log_coefficient_of_variation(predicted_points)

        else:
            loss = self.base_model(batch)
            if self.wandb: self.log('Validation Loss', loss)

        return loss

    def test_step(self):
        NotImplementedError()

    def predict_step(self, batch, batch_id, latent=False) -> Any:
        # NOTE make this code return the reconstruction is we have the full Point-MAE, 
        # and if we only have the encoder, make it return the latents. 
        return self.base_model(batch)

    def configure_optimizers(self):
        optimizer, scheduler = build_optim_scheme(self.base_model, self.cfg)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'Training Loss'}

    def log_pointclouds(self, reconstructed_pcd, unmasked_points, centers,
                        predicted_points, points, masked, latent):
        wandb.log({
            "Reconstructed Pointcloud": wandb.Object3D(reconstructed_pcd),
            "Predicted Points": wandb.Object3D(predicted_points)
        })

        if not self.unmasked_logged:
            wandb.log({
                "Unmasked Points": wandb.Object3D(unmasked_points),
                "Centers": wandb.Object3D(centers),
                "Masked Points": wandb.Object3D(masked),
                "Complete Input Pointcloud": wandb.Object3D(points)
            })
            self.unmasked_logged = True

    def log_coefficient_of_variation(self, predicted_points):
        groups = int(self.cfg.model.transformer_config.mask_ratio * self.cfg.model.num_group)
        points_per_group = int(self.cfg.model.group_size)
        reshaped_points = predicted_points.reshape(groups, points_per_group, 6)[:, :, 0:3]
        mcv = modified_coefficient_of_variation(reshaped_points)
        self.log("Difference between point patches", mcv)

    @classmethod
    def load_and_freeze_encoder(self, checkpoint_path: str, cfg, args, freeze: bool = True):
        """
        Load the pretrained Point_MAE model and freeze its encoder.

        :param checkpoint_path: Path to the checkpoint of the pretrained model.
        :param cfg: Configuration object.
        :return: Frozen encoder model.
        """
        # Load the pretrained model
        pretrained_point_mae = self.load_from_checkpoint(checkpoint_path, cfg=cfg, args=args)
        mae_encoder = pretrained_point_mae.base_model.MAE_encoder

        # Freeze the encoder
        if freeze:
            for param in mae_encoder.parameters():
                param.requires_grad = False

        return mae_encoder


class GAELightningModule(pl.LightningModule):
    """
    Lightning Module for the Group and Encode. 
    Extract latents and labels per patch using pretrained PointMAE.
    GAE = Group And Encode
    """

    def __init__(self, cfg, args, pretrained_encoder=None, base_type=False, mask=False) -> None:
        super(GAELightningModule, self).__init__()
        if pretrained_encoder is not None:
            # If a pretrained encoder is provided, use it
            self.base_model = model_builder(cfg.group_and_encode_model)
            self.base_model.MAE_encoder = pretrained_encoder
            print("PRETRAINED ENCODER LOADED SUCCESFULLY")
        else:
            raise ValueError("Pretrained encoder configuration has not been provided.")
        # set the masking ratio to 0, as the pretrained encoder != 0
        if mask == False:
            self.base_model.MAE_encoder.mask_ratio = 0.0
        print("FINAL MASK RATIO: ", self.base_model.MAE_encoder.mask_ratio)
        self.cfg = cfg
        self.wandb = args.wandb
        self.base_type = base_type

    def forward(self, points, labels, return_labels=False):
        return self.base_model(points, labels)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        The predict step for the GroupAndEncode model in a PyTorch Lightning workflow.
        This step processes the input point cloud data and returns encoded features and labels per patch.
        
        :param batch: The batch of data provided by the dataloader. Expected to be a tuple of points and labels.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the dataloader that provided the batch (useful if multiple dataloaders are used).
        :return: A tuple of encoded features, expanded labels, and the pairwise distances between all patches in a given sample.
        """
        if self.base_type:
            points, labels, base_types = batch
            encoded_features, majority_patch_labels = self.forward(points, labels)
            return encoded_features, majority_patch_labels, base_types
        else:
            points, labels = batch  # Unpack the batch of data
            # Forward pass through the base model to get encoded features and labels per patch
            encoded_features, majority_patch_labels = self.forward(points, labels)
            return encoded_features, majority_patch_labels


class GNNLightningModule(pl.LightningModule):
    def __init__(self, cfg, args):
        super(GNNLightningModule, self).__init__()
        self.model = model_builder(cfg.model)
        self.cfg = cfg
        self.args = args

        self.wandb = args.wandb
        self.all_y_pred = []
        self.all_y_true = []
        self.label_names = dict(pd.read_csv(cfg.labels)['Written Out Label'])
        self.model.print_summary()

    def forward(self, batch):
        # extract node feateure and labels
        x = batch.x
        y = batch.y

        # extract edges
        edge_index = batch.edge_index
        distance = batch.edge_attr

        loss, acc, pred, y = self.model(x, edge_index, distance, y)
        return loss, acc, pred, y

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.cfg.optimizer.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # TODO: remove manual batch size assignmet
        # TODO: ass wandb
        loss, acc, _, _ = self.forward(batch)
        self.log('Training Loss', loss, batch_size=self.cfg.batch_size)
        self.log('Training Accuracy', acc, batch_size=self.cfg.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: remove manual batch size assignmet
        loss, acc, pred, y = self.forward(batch)
        self.log('Validation Loss', loss, batch_size=self.cfg.batch_size)
        self.log('Validation Accuracy', acc, batch_size=self.cfg.batch_size)

        # Convert tensors to CPU and numpy format before storing
        (y_pred, y_true) = to_cpu(*(pred, y))
        # Append predictions and labels to lists instead of concatenating arrays
        self.all_y_true.append(y_true)
        self.all_y_pred.append(y_pred)

    def test_step(self, batch, batch_idx):
        loss, acc, _, _ = self.forward(batch)
        self.log('Test Loss', loss, batch_size=self.cfg.batch_size)
        self.log('Test Accuracy', acc, batch_size=self.cfg.batch_size)

    def on_validation_epoch_end(self):
        # Flatten lists of arrays into single arrays for reporting
        all_y_true_flat = np.concatenate(self.all_y_true)
        all_y_pred_flat = np.concatenate(self.all_y_pred)
        # Clear lists to free memory
        self.all_y_true.clear()
        self.all_y_pred.clear()

        # Generate classification report
        report = classification_report(all_y_true_flat, all_y_pred_flat, zero_division=0, output_dict=True)
        report_df = pd.DataFrame.from_dict(report).transpose()

        # Calculate total number of samples
        total_samples = len(all_y_true_flat)

        # Replace support numbers with percentages of the total
        report_df['support'] = report_df['support'].apply(lambda x: (x / total_samples) * 100)

        # Correctly map 'label name' including special entries
        def map_label_names(index):
            if index in ['accuracy', 'macro avg', 'weighted avg']:
                return index  # Keep these entries as is
            try:
                index = int(index)
                if index == 0:
                    return 'no label'
                else:
                    return self.label_names[index - 1]  # Adjust for 1-based indexing
            except (ValueError, IndexError):
                # Handle unexpected index values
                return 'unknown'

        report_df['label name'] = report_df.index.map(map_label_names)

        # Ensure 'label name' column is first
        cols = report_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        report_df = report_df[cols]
        print(report_df)

        # Log the classification report table to wandb
        if self.wandb:
            current_epoch = self.current_epoch  # PyTorch Lightning tracks the current epoch
            table_name = f" {wandb.run.name} - Classification Report Epoch {current_epoch}"
            wandb.log({table_name: wandb.Table(dataframe=report_df)})


class MLPLightningModule(pl.LightningModule):
    def __init__(self, cfg, args):
        super(MLPLightningModule, self).__init__()
        self.model = model_builder(cfg.model)
        self.cfg = cfg
        self.args = args

        self.wandb = args.wandb

    def forward(self, batch):
        x, y = batch
        balanced_acc, loss = self.model(x, y)
        return balanced_acc, loss

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.cfg.optimizer.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        balanced_acc, loss = self.forward(batch)
        self.log('Training Loss', loss, batch_size=self.cfg.batch_size)
        self.log('Training Accuracy', balanced_acc, batch_size=self.cfg.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        balanced_acc, loss = self.forward(batch)
        self.log('Validation Loss', loss, batch_size=self.cfg.batch_size)
        self.log('Validation Accuracy', balanced_acc, batch_size=self.cfg.batch_size)

    def test_step(self, batch, batch_idx):
        balanced_acc, loss = self.forward(batch)
        self.log('Test Loss', loss, batch_size=self.cfg.batch_size)
        self.log('Test Accuracy', balanced_acc, batch_size=self.cfg.batch_size)
