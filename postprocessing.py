import torch


def training_postprocess(outs, num_anchors, num_classes):

    for i in range(len(outs)):
        batch_size, _, grid_y, grid_x = outs[i].shape
        outs[i] = outs[i].view(batch_size, num_anchors, num_classes, grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()

    return outs



def validation_postprocess(outs, grid, anchor_grid, stride, num_classes, anchors):

    val_outs = []
    num_anchors = len(anchor_grid)
    train_outs = []

    for i in range(len(outs)):

        batch_size, _, grid_y, grid_x = outs[i].shape
        outs[i] = outs[i].view(batch_size, num_anchors, num_classes, grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()
        train_outs.append(outs[i])

        if grid[i].shape[2:4] != outs[i].shape[2:4]:

            grid[i], anchor_grid[i] = make_grid(anchors=anchors, num_anchors=num_anchors, stride=stride, grid_x=grid_x, grid_y=grid_y, i=i)

        y = outs[i].sigmoid()

        y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + grid[i]) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh

        val_outs.append(y.view(batch_size, -1, num_classes))

    return torch.cat(val_outs,1), train_outs


def make_grid(anchors, num_anchors, stride, grid_x=20, grid_y=20, i=0):
    
    device = anchors[i].device

    expanded_grid_y, expanded_grid_x = torch.meshgrid([torch.arange(grid_y).to(device), torch.arange(grid_x).to(device)])
    grid = torch.stack((expanded_grid_x, expanded_grid_y), 2).expand((1, num_anchors, grid_y, grid_x, 2)).float()
    anchor_grid = (anchors[i].clone() * stride[i]) \
        .view((1, num_anchors, 1, 1, 2)).expand((1, num_anchors, grid_y, grid_x, 2)).float()
    return grid, anchor_grid