import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t
        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * torch.tanh(c_new)
        
        return h_new, c_new


class ConvLSTM_Model(nn.Module):
    def __init__(self, num_layers, num_hidden, patch_size, filter_size, stride, layer_norm):
        super(ConvLSTM_Model, self).__init__()
        _, C, H, W = 64, 32

        self.frame_channel = patch_size * patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        
        height = H // patch_size
        width = W // patch_size
        self.MSE_criterion = nn.MSELoss()

        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            self.encoder_cells.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, 
                           filter_size, stride, layer_norm)
            )

        self.forecast_cells = nn.ModuleList()
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            self.forecast_cells.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, 
                           filter_size, stride, layer_norm)
            )

        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, **kwargs):
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        h_t_encoder = []
        c_t_encoder = []
        
        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(device)
            h_t_encoder.append(zeros)
            c_t_encoder.append(zeros)

        for t in range(frames.shape[1]):
            net = frames[:, t]

            for i in range(self.num_layers):
                input_tensor = net if i == 0 else h_t_encoder[i-1]
                h_t_encoder[i], c_t_encoder[i] = self.encoder_cells[i](
                    input_tensor, h_t_encoder[i], c_t_encoder[i])

        h_t_forecast = h_t_encoder.copy()
        c_t_forecast = c_t_encoder.copy()

        x_pred = self.conv_last(h_t_forecast[self.num_layers - 1])

        for i in range(self.num_layers):
            input_tensor = x_pred if i == 0 else h_t_forecast[i-1]
            h_t_forecast[i], c_t_forecast[i] = self.forecast_cells[i](
                input_tensor, h_t_forecast[i], c_t_forecast[i])

        forecast_plus_6h = self.conv_last(h_t_forecast[self.num_layers - 1])

        forecast_plus_6h = forecast_plus_6h.permute(0, 2, 3, 1).contiguous()
        
        
        return forecast_plus_6h


    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info