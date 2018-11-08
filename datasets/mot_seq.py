import numpy as np
import os
import torch.utils.data as data
from scipy.misc import imread

from utils.io import read_mot_results, unzip_objs


"""
labels={'ped', ...			% 1
'person_on_vhcl', ...	% 2
'car', ...				% 3
'bicycle', ...			% 4
'mbike', ...			% 5
'non_mot_vhcl', ...		% 6
'static_person', ...	% 7
'distractor', ...		% 8
'occluder', ...			% 9
'occluder_on_grnd', ...		%10
'occluder_full', ...		% 11
'reflection', ...		% 12
'crowd' ...			% 13
};
"""


# def read_mot_results(filename, is_gt=False):
#     labels = {1, -1}
#     targets = dict()
#     if os.path.isfile(filename):
#         with open(filename, 'r') as f:
#             for line in f.readlines():
#                 linelist = line.split(',')
#                 if len(linelist) < 7:
#                     continue
#                 fid = int(linelist[0])
#                 targets.setdefault(fid, list())
#
#                 if is_gt and ('MOT16-' in filename or 'MOT17-' in filename):
#                     label = int(float(linelist[-2])) if len(linelist) > 7 else -1
#                     if label not in labels:
#                         continue
#                 tlwh = tuple(map(float, linelist[2:7]))
#                 target_id = int(linelist[1])
#
#                 targets[fid].append((tlwh, target_id))
#
#     return targets


class MOTSeq(data.Dataset):
    def __init__(self, root, det_root, seq_name, min_height, min_det_score):
        self.root = root
        self.seq_name = seq_name
        self.min_height = min_height
        self.min_det_score = min_det_score

        self.im_root = os.path.join(self.root, self.seq_name, 'img1')
        self.im_names = sorted([name for name in os.listdir(self.im_root) if os.path.splitext(name)[-1] == '.jpg'])

        if det_root is None:
            self.det_file = os.path.join(self.root, self.seq_name, 'det', 'det.txt')
        else:
            self.det_file = os.path.join(det_root, '{}.txt'.format(self.seq_name))
        self.dets = read_mot_results(self.det_file, is_gt=False, is_ignore=False)

        self.gt_file = os.path.join(self.root, self.seq_name, 'gt', 'gt.txt')
        if os.path.isfile(self.gt_file):
            self.gts = read_mot_results(self.gt_file, is_gt=True, is_ignore=False)
        else:
            self.gts = None

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, i):
        im_name = os.path.join(self.im_root, self.im_names[i])
        # im = cv2.imread(im_name)
        im = imread(im_name)  # rgb
        im = im[:, :, ::-1]  # bgr

        frame = i + 1
        dets = self.dets.get(frame, [])
        tlwhs, _, scores = unzip_objs(dets)
        scores = np.asarray(scores)

        keep = (tlwhs[:, 3] >= self.min_height) & (scores > self.min_det_score)
        tlwhs = tlwhs[keep]
        scores = scores[keep]

        if self.gts is not None:
            gts = self.gts.get(frame, [])
            gt_tlwhs, gt_ids, _ = unzip_objs(gts)
        else:
            gt_tlwhs, gt_ids = None, None

        return im, tlwhs, scores, gt_tlwhs, gt_ids


def collate_fn(data):
    return data[0]


def get_loader(root, det_root, name, min_height=0, min_det_score=-np.inf, num_workers=3):
    dataset = MOTSeq(root, det_root, name, min_height, min_det_score)

    data_loader = data.DataLoader(dataset, 1, False, num_workers=num_workers, collate_fn=collate_fn)

    return data_loader
