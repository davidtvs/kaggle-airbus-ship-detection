import torch.nn as nn
import torch


class ComboNet(nn.Module):
    def __init__(self, clf_net, seg_net, clf_output_fn=None, seg_output_fn=None):
        super().__init__()
        self.clf_net = clf_net
        self.seg_net = seg_net
        self.clf_output_fn = clf_output_fn
        self.seg_output_fn = seg_output_fn

    def forward(self, x):
        clf_out = self.clf_net(x)
        if self.clf_output_fn is not None:
            clf_out = self.clf_output_fn(clf_out)

        has_ship_mask = torch.eq(clf_out, torch.Tensor([1]).to(x.device)).squeeze()
        out_size = (x.size(0), self.seg_net.num_classes, x.size(2), x.size(3))
        out = torch.zeros(out_size, device=x.device)

        if has_ship_mask.any():
            seg_out = self.seg_net(x[has_ship_mask])
            if self.seg_output_fn is not None:
                seg_out = self.seg_output_fn(seg_out)
                out[has_ship_mask] = seg_out

        return out
