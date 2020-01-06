import os.path
from itertools import chain

import torch
from PIL import Image

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid

class OuluDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.source_view = opt.source_view
        self.target_view = opt.target_view
        self.dir_A = os.path.join(opt.dataroot, str(opt.source_view) + '/' + opt.phase + '/')
        self.dir_B = os.path.join(opt.dataroot, str(opt.target_view) + '/' + opt.phase + '/')
        self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        self.B_paths = sorted(make_grouped_dataset(self.dir_B))
        self.A_paths = list(chain.from_iterable(self.A_paths))
        self.B_paths = list(chain.from_iterable(self.B_paths))
        self.init_frame_idx(self.A_paths)
        self.n_of_seqs = len(self.A_paths)                 # number of sequences to train


    def __getitem__(self, index):
        A, B, I, seq_idx = self.update_frame_idx(self.A_paths, index)
        A_path = self.A_paths[seq_idx]
        B_path = self.B_paths[seq_idx]
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(A_path), self.frame_idx)

        B_img = Image.open(B_path).convert('RGB')
        params = get_img_params(self.opt, B_img.size)
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = transform_scaleB

        Ai = self.get_image(A_path, transform_scaleA)
        Bi = self.get_image(B_path, transform_scaleB)

        A = concat_frame(A,Ai,n_frames_total)
        B = concat_frame(B,Bi,n_frames_total)
        self.seq_idx+=1
        return_list = {'A': A, 'B': B, 'inst': 0, 'A_path': A_path, 'B_paths': B_path, 'change_seq': False}
        return return_list

    def get_image(self, A_path, transform_scaleA):
        A_img = Image.open(A_path)
        A_scaled = transform_scaleA(A_img)
        # if is_label:
        #     A_scaled *= 255.0
        return A_scaled

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'OuluDataset'