import torch


def normalize_data(data, channel_mapping):
    norm_params = {}

    for key, val in channel_mapping['key_to_channels'].items():
        if len(val) == 0:
            mean = data[:, :, :, :, val[0]].float().mean().item()
            std = data[:, :, :, :, val[0]].float().std().item() + 1e-8
            data[:, :, :, :, val[0]] = (data[:, :, :, :, val[0]] - mean) / std
            norm_params[key] = {"mean": mean, "std": std}
        else:
            means = []
            stds = []
            for level in val:
                mean = data[:, :, :, :, level].float().mean().item()
                std = data[:, :, :, :, level].float().std().item() + 1e-8
                data[:, :, :, :, level] = (data[:, :, :, :, level] - mean) / std
                means.append(mean)
                stds.append(std)
            norm_params[key] = {"mean": means, "std": stds}

    return data, norm_params

def collate_fn(dataset_items: list[dict]):
    batch_tensors = []
    sorted_keys = sorted(dataset_items[0].keys())
    channel_mapping = {
        'key_to_channels': {},
    }
    
    current_channel = 0
    
    for data_dict in dataset_items:
        dict_tensors = []

        if len(batch_tensors) == 0:
            for key in sorted_keys:
                tensor = data_dict[key]
                if len(tensor.shape) == 4:
                    num_pressures = tensor.shape[1]
                    channels = list(range(current_channel, current_channel + num_pressures))
                    channel_mapping['key_to_channels'][key] = channels
                    
                    current_channel += num_pressures
                elif len(tensor.shape) == 3:
                    channel_mapping['key_to_channels'][key] = [current_channel]
                    current_channel += 1

        for key in sorted_keys:
            tensor = data_dict[key]
            if len(tensor.shape) == 4:
                reshaped_tensor = tensor.permute(0, 2, 3, 1)
            elif len(tensor.shape) == 3:
                reshaped_tensor = tensor.unsqueeze(-1)
            
            dict_tensors.append(reshaped_tensor)

        combined_tensor = torch.cat(dict_tensors, dim=-1)
        batch_tensors.append(combined_tensor)

    final_tensor = torch.stack(batch_tensors, dim=0).permute(0, 1, 3, 2, 4)
    final_tensor, norm_params = normalize_data(final_tensor, channel_mapping)
    return {"batch":final_tensor, "mapping": channel_mapping, "norm_params": norm_params}
