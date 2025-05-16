from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch
from src.loss import PhysicallyInformedLoss, HumidityPhysicallyInformedLoss, SurfacePressurePhysicallyInformedLoss, TemperaturePhysicallyInformedLoss
from collections import defaultdict
import pandas as pd
import os

SAVE_DIR = "/home/andrewut/diplom/Global-climate-forecasting-using-pinn-loss/saved"

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, epoch:int, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()
        outputs = self.model(batch['batch'])
        all_losses = self.criterion(outputs[:, 12:], batch['batch'][:, 12:, :, :, :], batch)
        if not isinstance(all_losses, dict):
            batch.update({"loss": all_losses})
        else:
            batch.update(all_losses)
        batch.update({"output_for_pinn_metric": outputs[:, 12:], "ground_truth_for_pinn_metric":batch['batch'][:, 12:, :, :, :]})

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(outputs[:, 12:], batch['batch'][:, 12:12+12-1, :, :, :]))
        return batch

    def _log_batch(self, batch_idx, batch,epoch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if not os.path.exists(os.path.join(SAVE_DIR, self.writer.run_name, "tables")):
            os.mkdir(os.path.join(SAVE_DIR, self.writer.run_name, "tables"))
        self._log_vorticity_pin_loss(batch, epoch, mode)
        self._log_surface_pin_loss(batch, epoch, mode)
        self._log_temperature_pin_loss(batch, epoch, mode)
        self._log_humidity_pin_loss(batch, epoch, mode)
    
    def _denormalize_data(self, data, channel_mapping, norm_params):
        for key, val in channel_mapping['key_to_channels'].items():
            if len(val) == 0:
                mean = norm_params[key]['mean']
                std = norm_params[key]['std']
                data[:, :, val[0]] = data[:, :, val[0]] * std + mean
                norm_params[key] = {"mean": mean, "std": std}
            else:
                for level in val:
                    mean = norm_params[key]['mean'][level - min(val)]
                    std = norm_params[key]['std'][level - min(val)]
                    data[:, :, level] = data[:, :, level]*std + mean
        
        return data
    
    def _log_vorticity_pin_loss(self, batch:dict, epoch, part):
        metric = PhysicallyInformedLoss()
        pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

        output: torch.Tensor = batch['output_for_pinn_metric'][0]
        # denormalized_output = self._denormalize_data(output.unsqueeze(0).permute(0, 1, 4, 2, 3), batch['mapping'], batch['norm_params'])
        calculated_metric_for_output = metric.calculate_residuals_from_tensor(output.unsqueeze(0).permute(0, 1, 4, 2, 3), dx=5.625, dy=5.625)
        data = defaultdict(list)
        data['time'] = [f"+{6 * (i + 1)}" for i in range(calculated_metric_for_output[0].shape[0])]

        calculated_metric_for_output = torch.abs(calculated_metric_for_output).mean(dim=[-2, -1])
        for moment_time in range(calculated_metric_for_output[0].shape[0]):
            for i, level in enumerate(pressure_levels):
                data[f'pressure_{level}_hPa'].append(calculated_metric_for_output[0][moment_time][i].item())

        self.writer.add_table("output_vorticity_pinn_loss_metric_per_pressure_levels", pd.DataFrame(data))
        pd.DataFrame(data).to_excel(os.path.join(SAVE_DIR, self.writer.run_name,"tables", f"output_vorticity_pinn_loss_metric_per_pressure_levels_{part}_{epoch}.xlsx"), index=False)

        ground_truth: torch.Tensor = batch['ground_truth_for_pinn_metric'][0]
        # denormalized_ground_truth = self._denormalize_data(ground_truth.unsqueeze(0).permute(0, 1, 4, 2, 3), batch['mapping'], batch['norm_params'])
        calculated_metric_for_ground_truth = metric.calculate_residuals_from_tensor(ground_truth.unsqueeze(0).permute(0, 1, 4, 2, 3), dx=5.625, dy=5.625)
        data = defaultdict(list)
        data['time'] = [f"+{6 * (i + 1)}" for i in range(calculated_metric_for_ground_truth[0].shape[0])]

        calculated_metric_for_ground_truth = torch.abs(calculated_metric_for_ground_truth).mean(dim=[-2, -1])
        for moment_time in range(calculated_metric_for_ground_truth[0].shape[0]):
            for i, level in enumerate(pressure_levels):
                data[f'pressure_{level}_hPa'].append(calculated_metric_for_ground_truth[0][moment_time][i].item())

        self.writer.add_table("ground_truth_vorticity_pinn_loss_metric_per_pressure_levels", pd.DataFrame(data))
        pd.DataFrame(data).to_excel(os.path.join(SAVE_DIR, self.writer.run_name,"tables", f"ground_truth_vorticity_pinn_loss_metric_per_pressure_levels_{part}_{epoch}.xlsx"), index=False)
    
    def _log_humidity_pin_loss(self, batch:dict, epoch, part):
        metric = HumidityPhysicallyInformedLoss()
        pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

        output: torch.Tensor = batch['output_for_pinn_metric'][0]
        # denormalized_output = self._denormalize_data(output.unsqueeze(0).permute(0, 1, 4, 2, 3), batch['mapping'], batch['norm_params'])
        calculated_metric_for_output = metric.calculate_residuals_from_tensor(output.unsqueeze(0).permute(0, 1, 4, 2, 3), dx=5.625, dy=5.625)
        data = defaultdict(list)
        data['time'] = [f"+{6 * (i + 1)}" for i in range(calculated_metric_for_output[0].shape[0])]

        calculated_metric_for_output = torch.abs(calculated_metric_for_output).mean(dim=[-2, -1])
        for moment_time in range(calculated_metric_for_output[0].shape[0]):
            for i, level in enumerate(pressure_levels):
                data[f'pressure_{level}_hPa'].append(calculated_metric_for_output[0][moment_time][i].item())

        self.writer.add_table("output_humidity_pinn_loss_metric_per_pressure_levels", pd.DataFrame(data))
        pd.DataFrame(data).to_excel(os.path.join(SAVE_DIR, self.writer.run_name,"tables", f"output_humidity_pinn_loss_metric_per_pressure_levels_{part}_{epoch}.xlsx"), index=False)

        ground_truth: torch.Tensor = batch['ground_truth_for_pinn_metric'][0]
        # denormalized_ground_truth = self._denormalize_data(ground_truth.unsqueeze(0).permute(0, 1, 4, 2, 3), batch['mapping'], batch['norm_params'])
        calculated_metric_for_ground_truth = metric.calculate_residuals_from_tensor(ground_truth.unsqueeze(0).permute(0, 1, 4, 2, 3), dx=5.625, dy=5.625)
        data = defaultdict(list)
        data['time'] = [f"+{6 * (i + 1)}" for i in range(calculated_metric_for_ground_truth[0].shape[0])]

        calculated_metric_for_ground_truth = torch.abs(calculated_metric_for_ground_truth).mean(dim=[-2, -1])
        for moment_time in range(calculated_metric_for_ground_truth[0].shape[0]):
            for i, level in enumerate(pressure_levels):
                data[f'pressure_{level}_hPa'].append(calculated_metric_for_ground_truth[0][moment_time][i].item())

        self.writer.add_table("ground_truth_humidity_pinn_loss_metric_per_pressure_levels", pd.DataFrame(data))
        pd.DataFrame(data).to_excel(os.path.join(SAVE_DIR, self.writer.run_name,"tables", f"ground_truth_humidity_pinn_loss_metric_per_pressure_levels_{part}_{epoch}.xlsx"), index=False)
    
    def _log_surface_pin_loss(self, batch:dict, epoch, part):
        metric = SurfacePressurePhysicallyInformedLoss()
        pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

        output: torch.Tensor = batch['output_for_pinn_metric'][0]
        # denormalized_output = self._denormalize_data(output.unsqueeze(0).permute(0, 1, 4, 2, 3), batch['mapping'], batch['norm_params'])
        calculated_metric_for_output = metric.calculate_residuals_from_tensor(output.unsqueeze(0).permute(0, 1, 4, 2, 3), dx=5.625, dy=5.625)
        data = defaultdict(list)
        data['time'] = [f"+{6 * (i + 1)}" for i in range(calculated_metric_for_output[0].shape[0])]

        calculated_metric_for_output = torch.abs(calculated_metric_for_output).mean(dim=[-2, -1])
        for moment_time in range(calculated_metric_for_output[0].shape[0]):
            data['surface_pressure'].append(calculated_metric_for_output[0][moment_time].item())

        self.writer.add_table("output_surface_pinn_loss_metric_per_pressure_levels", pd.DataFrame(data))
        pd.DataFrame(data).to_excel(os.path.join(SAVE_DIR, self.writer.run_name,"tables", f"output_surface_pinn_loss_metric_per_pressure_levels_{part}_{epoch}.xlsx"), index=False)

        ground_truth: torch.Tensor = batch['ground_truth_for_pinn_metric'][0]
        # denormalized_ground_truth = self._denormalize_data(ground_truth.unsqueeze(0).permute(0, 1, 4, 2, 3), batch['mapping'], batch['norm_params'])
        calculated_metric_for_ground_truth = metric.calculate_residuals_from_tensor(ground_truth.unsqueeze(0).permute(0, 1, 4, 2, 3), dx=5.625, dy=5.625)
        data = defaultdict(list)
        data['time'] = [f"+{6 * (i + 1)}" for i in range(calculated_metric_for_ground_truth[0].shape[0])]

        calculated_metric_for_ground_truth = torch.abs(calculated_metric_for_ground_truth).mean(dim=[-2, -1])
        for moment_time in range(calculated_metric_for_ground_truth[0].shape[0]):
            data['surface_pressure'].append(calculated_metric_for_ground_truth[0][moment_time].item())

        self.writer.add_table("ground_truth_surface_pinn_loss_metric_per_pressure_levels", pd.DataFrame(data))
        pd.DataFrame(data).to_excel(os.path.join(SAVE_DIR, self.writer.run_name,"tables", f"ground_truth_surface_pinn_loss_metric_per_pressure_levels_{part}_{epoch}.xlsx"), index=False)
    
    def _log_temperature_pin_loss(self, batch:dict, epoch, part):
        metric = TemperaturePhysicallyInformedLoss()
        pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

        output: torch.Tensor = batch['output_for_pinn_metric'][0]
        # denormalized_output = self._denormalize_data(output.unsqueeze(0).permute(0, 1, 4, 2, 3), batch['mapping'], batch['norm_params'])
        calculated_metric_for_output = metric.calculate_residuals_from_tensor(output.unsqueeze(0).permute(0, 1, 4, 2, 3), dx=5.625, dy=5.625)
        data = defaultdict(list)
        data['time'] = [f"+{6 * (i + 1)}" for i in range(calculated_metric_for_output[0].shape[0])]

        calculated_metric_for_output = torch.abs(calculated_metric_for_output).mean(dim=[-2, -1])
        for moment_time in range(calculated_metric_for_output[0].shape[0]):
            for i, level in enumerate(pressure_levels):
                data[f'pressure_{level}_hPa'].append(calculated_metric_for_output[0][moment_time][i].item())

        self.writer.add_table("output_temperature_pinn_loss_metric_per_pressure_levels", pd.DataFrame(data))
        pd.DataFrame(data).to_excel(os.path.join(SAVE_DIR, self.writer.run_name, "tables", f"output_temperature_pinn_loss_metric_per_pressure_levels_{part}_{epoch}.xlsx"), index=False)

        ground_truth: torch.Tensor = batch['ground_truth_for_pinn_metric'][0]
        # denormalized_ground_truth = self._denormalize_data(ground_truth.unsqueeze(0).permute(0, 1, 4, 2, 3), batch['mapping'], batch['norm_params'])
        calculated_metric_for_ground_truth = metric.calculate_residuals_from_tensor(ground_truth.unsqueeze(0).permute(0, 1, 4, 2, 3), dx=5.625, dy=5.625)
        data = defaultdict(list)
        data['time'] = [f"+{6 * (i + 1)}" for i in range(calculated_metric_for_ground_truth[0].shape[0])]

        calculated_metric_for_ground_truth = torch.abs(calculated_metric_for_ground_truth).mean(dim=[-2, -1])
        for moment_time in range(calculated_metric_for_ground_truth[0].shape[0]):
            for i, level in enumerate(pressure_levels):
                data[f'pressure_{level}_hPa'].append(calculated_metric_for_ground_truth[0][moment_time][i].item())

        self.writer.add_table("ground_truth_temperature_pinn_loss_metric_per_pressure_levels", pd.DataFrame(data))
        pd.DataFrame(data).to_excel(os.path.join(SAVE_DIR, self.writer.run_name,"tables", f"ground_truth_temperature_pinn_loss_metric_per_pressure_levels_{part}_{epoch}.xlsx"), index=False)
