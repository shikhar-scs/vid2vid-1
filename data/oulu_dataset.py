import os.path

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
        check_path_valid(self.A_paths, self.B_paths)
        self.init_frame_idx(self.A_paths)


    def __getitem__(self, index):
        A, B, I, seq_idx = self.update_frame_idx(self.A_paths, index)
        print(index)
        A_paths = self.A_paths[seq_idx]
        B_paths = self.B_paths[seq_idx]
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(A_paths), self.frame_idx)

        B_img = Image.open(B_paths[start_idx]).convert('RGB')
        params = get_img_params(self.opt, B_img.size)
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = transform_scaleB

        frame_range = list(range(n_frames_total)) if self.A is None else [self.opt.n_frames_G-1]
        for i in frame_range:
            A_path = A_paths[start_idx + i * t_step]
            B_path = B_paths[start_idx + i * t_step]
            Ai= self.get_image(A_path, transform_scaleA)
            Bi= self.get_image(B_path, transform_scaleB)
            A = concat_frame(A, Ai, n_frames_total)
            B = concat_frame(B, Bi, n_frames_total)

        if not self.opt.isTrain:
            self.A, self.B, self.I = A, B, I
            self.frame_idx += 1

        change_seq = False if self.opt.isTrain else self.change_seq
        return_list = {'A': A, 'B': B, 'inst': 0, 'A_path': A_path, 'B_paths': B_path, 'change_seq': change_seq}

        return return_list

    def get_image(self, A_path, transform_scaleA):
        A_img = Image.open(A_path)
        A_scaled = transform_scaleA(A_img)
        return A_scaled

    def __len__(self):
        if self.opt.isTrain:
            return len(self.A_paths)
        else:
            return sum(self.frames_count)

    def name(self):
        return 'OuluDataset'