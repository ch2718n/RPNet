import torch
import torch.nn.functional as F


def nearest_completion(x, k=3):
    x_pad = F.pad(x, (k // 2, k // 2, k // 2, k // 2))
    x_neighbor = [x_pad[:, :, i // k:x_pad.size(2) + i // k - k + 1, i % k:x_pad.size(3) + i % k - k + 1]
                  for i in range(k ** 2) if i != k ** 2 // 2]
    x_neighbor = torch.stack(x_neighbor, dim=0)
    xyz = x[:, :3, :, :]
    xyz_pad = F.pad(xyz, (k // 2, k // 2, k // 2, k // 2))
    mask = (xyz_pad != 0).float()
    conv_weight = torch.ones((3, 1, k, k), device=x.device)
    center = F.conv2d(xyz_pad, weight=conv_weight, groups=3) / F.conv2d(mask, weight=conv_weight, groups=3)

    distance = [
        torch.sum(
            (xyz_pad[:, :, i // k:xyz_pad.size(2) + i // k - k + 1, i % k:xyz_pad.size(3) + i % k - k + 1] - center)
            ** 2, dim=1, keepdim=True) for i in range(k ** 2) if i != k ** 2 // 2]
    distance = torch.stack(distance, dim=0).repeat((1, 1, x.shape[1], 1, 1))
    nearest_index = torch.argmin(distance, dim=0, keepdim=True)
    nearest_value = torch.gather(x_neighbor, dim=0, index=nearest_index)
    nearest_value = nearest_value.squeeze(0)
    x = x + (x == 0) * nearest_value

    return x
