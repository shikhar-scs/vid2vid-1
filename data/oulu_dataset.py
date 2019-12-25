import os.path
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
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + '/phrase/' + str(opt.source_view))
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + '/phrase/' + str(opt.target_view))
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '/digit/' + str(opt.source_view))
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '/digit/' + str(opt.target_view))

        self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        self.B_paths = sorted(make_grouped_dataset(self.dir_B))
        check_path_valid(self.A_paths, self.B_paths)

        self.n_of_seqs = len(self.A_paths)  # number of sequences to train
        # self.seq_len_max = max([len(A) for A in self.A_paths])
        self.seq_len_max = 30
        self.n_frames_total = self.opt.n_frames_total  # current number of frames to train in a single iteration

    def __getitem__(self, index):
        tG = self.opt.n_frames_G
        A_paths = self.A_paths[index % self.n_of_seqs]
        B_paths = self.B_paths[index % self.n_of_seqs]

        # setting parameters
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(A_paths), index)

        # setting transformers
        B_img = Image.open(B_paths[start_idx]).convert('RGB')
        params = get_img_params(self.opt, B_img.size)
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = transform_scaleB

        # read in images
        A = B = 0
        for i in range(n_frames_total):
            A_path = A_paths[start_idx + i * t_step]
            B_path = B_paths[start_idx + i * t_step]
            Ai = self.get_image(A_path, transform_scaleA)
            Bi = self.get_image(B_path, transform_scaleB)

            A = Ai if i == 0 else torch.cat([A, Ai], dim=0)
            B = Bi if i == 0 else torch.cat([B, Bi], dim=0)

        return_list = {'A': A, 'B': B, 'A_path': A_path, 'B_paths': B_path}
        # torch.save(A, 'A_Tensor.pt')
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