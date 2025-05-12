import torch

def collate_fn(dataset_items: list[dict]):
    batch_tensors = []
    sorted_keys = sorted(dataset_items[0].keys())

    channel_mapping = {
        'key_to_channels': {},
        'channel_to_info': {},
        'total_channels': 0
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
                    
                    for p_idx, channel in enumerate(channels):
                        channel_mapping['channel_to_info'][channel] = {
                            'key': key,
                            'pressure_level': p_idx
                        }
                    
                    current_channel += num_pressures
                elif len(tensor.shape) == 3:
                    channel_mapping['key_to_channels'][key] = [current_channel]
                    channel_mapping['channel_to_info'][current_channel] = {
                        'key': key,
                        'pressure_level': None
                    }
                    current_channel += 1
            
            channel_mapping['total_channels'] = current_channel

        for key in sorted_keys:
            tensor = data_dict[key]
            if len(tensor.shape) == 4:
                reshaped_tensor = tensor.permute(0, 2, 3, 1)
            elif len(tensor.shape) == 3:
                reshaped_tensor = tensor.unsqueeze(-1)
            else:
                raise ValueError(f"Неподдерживаемая размерность тензора для ключа {key}: {tensor.shape}")
            
            dict_tensors.append(reshaped_tensor)

        combined_tensor = torch.cat(dict_tensors, dim=-1)
        batch_tensors.append(combined_tensor)

    final_tensor = torch.stack(batch_tensors, dim=0)
    
    return {"batch":final_tensor, "mapping": channel_mapping}
