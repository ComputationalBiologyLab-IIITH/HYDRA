import numpy as np

import torch
import torch.optim as optim
import pytorch_lightning as pl

from torch_scatter import scatter_mean

from utils.evaluation import atom_num
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]])
    all_step_v = [
        np.stack(step_v) for step_v in all_step_v
    ]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


# Target Aware Residue Prediction
class TARP(pl.LightningModule):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config
        self.model = ScorePosNet3D(
            self.config.model,
            protein_atom_feature_dim=protein_atom_feature_dim,
            ligand_atom_feature_dim=ligand_atom_feature_dim,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config.train.optimizer.lr,
            weight_decay=self.config.train.optimizer.weight_decay,
            betas=(
                self.config.train.optimizer.beta1,
                self.config.train.optimizer.beta2,
            ),
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.config.train.scheduler.factor,
            patience=self.config.train.scheduler.patience,
            min_lr=self.config.train.scheduler.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": self.config.train.val_freq,
            },
        }

    def training_step(self, batch, batch_idx):
        protein_noise = (
            torch.randn_like(batch.protein_pos) * self.config.train.pos_noise_std
        )
        gt_protein_pos = batch.protein_pos + protein_noise

        results = self.model.get_diffusion_loss(
            protein_pos=gt_protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,
            ligand_pos=batch.ligand_pos,
            ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch,
        )

        loss, loss_pos, loss_v = results["loss"], results["loss_pos"], results["loss_v"]
        loss = loss / self.config.train.n_acc_batch

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_loss_pos", loss_pos, sync_dist=True)
        self.log("train_loss_v", loss_v, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch.num_graphs

        sum_loss, sum_loss_pos, sum_loss_v = 0, 0, 0
        for t in torch.linspace(0, self.model.num_timesteps - 1, 10, dtype=torch.long):
            time_step = torch.tensor([t] * batch_size)

            results = self.model.get_diffusion_loss(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,
                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,
                time_step=time_step,
            )

            loss, loss_pos, loss_v = (
                results["loss"],
                results["loss_pos"],
                results["loss_v"],
            )
            sum_loss += float(loss)
            sum_loss_pos += float(loss_pos)
            sum_loss_v += float(loss_v)
        sum_loss /= 10
        sum_loss_pos /= 10
        sum_loss_v /= 10

        self.log("val_loss", sum_loss, batch_size=batch_size, sync_dist=True)
        self.log("val_loss_pos", sum_loss_pos, batch_size=batch_size, sync_dist=True)
        self.log("val_loss_v", sum_loss_v, batch_size=batch_size, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        n_data = len(batch)

        batch_protein = batch.protein_element_batch

        # sample_num_atoms: prior
        pocket_size = atom_num.get_space_size(batch.protein_pos)
        ligand_num_atoms = [
            atom_num.sample_atom_num(pocket_size) for _ in range(n_data)
        ]
        batch_ligand = torch.repeat_interleave(
            torch.arange(n_data, device=batch_protein.device),
            torch.tensor(ligand_num_atoms, device=batch_protein.device),
        )

        center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
        batch_center_pos = center_pos[batch_ligand]
        init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

        uniform_logits = torch.zeros(
            len(batch_ligand),
            self.model.num_classes,
            device=batch_protein.device,
        )
        init_ligand_v = log_sample_categorical(uniform_logits)

        r = self.model.sample_diffusion(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch_protein,
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=batch_ligand,
            num_steps=self.config.sample.num_steps,
            pos_only=self.config.sample.pos_only,
            center_pos_mode=self.config.sample.center_pos_mode,
        )

        ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = (
            r["pos"],
            r["v"],
            r["pos_traj"],
            r["v_traj"],
        )

        # unbatch pos
        ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
        ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
        all_pred_pos = [
            ligand_pos_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]]
            for k in range(n_data)
        ]  # num_samples * [num_atoms_i, 3]

        all_step_pos = [[] for _ in range(n_data)]
        for p in ligand_pos_traj:  # step_i
            p_array = p.cpu().numpy().astype(np.float64)
            for k in range(n_data):
                all_step_pos[k].append(
                    p_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]]
                )
        all_step_pos = [
            np.stack(step_pos) for step_pos in all_step_pos
        ]  # num_samples * [num_steps, num_atoms_i, 3]
        all_pred_pos_traj = list(all_step_pos)

        # unbatch v
        ligand_v_array = ligand_v.cpu().numpy()
        all_pred_v = [
            ligand_v_array[ligand_cum_atoms[k] : ligand_cum_atoms[k + 1]]
            for k in range(n_data)
        ]

        all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
        all_pred_v_traj = list(all_step_v)

        return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj
