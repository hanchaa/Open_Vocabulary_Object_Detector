import torch
import json


def load_class_freq(path='datasets/metadata/lvis_v1_train_cat_info.json', freq_weight=1.0):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.float() ** freq_weight
    return freq_weight


def get_fed_loss_inds(gt_classes, num_sample_cats, num_classes, weight=None):
    appeared = torch.unique(gt_classes)
    prob = appeared.new_ones(num_classes + 1).float()
    prob[-1] = 0

    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:num_classes] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])

    return appeared
