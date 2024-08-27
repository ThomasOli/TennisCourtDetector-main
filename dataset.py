import cv2
import numpy as np
from utils import draw_umich_gaussian, line_intersection, is_point_in_image

class courtDataset(Dataset):

    def __init__(self, mode, input_height=720, input_width=1280, scale=2, hp_radius=55):
        self.mode = mode
        assert mode in ['train', 'val'], 'incorrect mode'
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = int(input_height/scale)
        self.output_width = int(input_width/scale)
        self.num_joints = 14
        self.hp_radius = hp_radius
        self.scale = scale

        self.path_dataset = './data'
        self.path_images = os.path.join(self.path_dataset, 'images')
        with open(os.path.join(self.path_dataset, 'data_{}.json'.format(mode)), 'r') as f:
            self.data = json.load(f)
        print('mode = {}, len = {}'.format(mode, len(self.data)))

    def __getitem__(self, index):
        img_name = self.data[index]['id'] + '.png'
        kps = self.data[index]['kps']
        img = cv2.imread(os.path.join(self.path_images, img_name))
        img = cv2.resize(img, (self.output_width, self.output_height))
        inp = (img.astype(np.float32) / 255.)
        inp = np.rollaxis(inp, 2, 0)

        # Identify points that are within the image bounds
        valid_points = [(kp[0], kp[1]) for kp in kps if is_point_in_image(kp[0], kp[1], self.input_width, self.input_height)]

        # If not all key points are valid, calculate homography
        if len(valid_points) < self.num_joints:
            # Define the destination points from the reference court model (you may need to adapt these points)
            dst_points = np.array([[x, y] for (x, y) in valid_points], dtype='float32')
            # Example src points from the reference model (you should have this data)
            src_points = np.array([[x, y] for (x, y) in some_reference_points], dtype='float32')  # Adapt this

            if len(dst_points) >= 4:  # Homography requires at least 4 points
                H, _ = cv2.findHomography(src_points, dst_points)
                for i, kp in enumerate(kps):
                    if not is_point_in_image(kp[0], kp[1], self.input_width, self.input_height):
                        # Project the missing point using homography
                        pt = np.array([kp[0], kp[1], 1]).reshape(-1, 1)
                        projected_pt = np.dot(H, pt)
                        kps[i][0] = projected_pt[0] / projected_pt[2]
                        kps[i][1] = projected_pt[1] / projected_pt[2]

        hm_hp = np.zeros((self.num_joints+1, self.output_height, self.output_width), dtype=np.float32)
        draw_gaussian = draw_umich_gaussian

        for i in range(len(kps)):
            if kps[i][0] >=0 and kps[i][0] <= self.input_width and kps[i][1] >=0 and kps[i][1] <= self.input_height:
                x_pt_int = int(kps[i][0]/self.scale)
                y_pt_int = int(kps[i][1]/self.scale)
                draw_gaussian(hm_hp[i], (x_pt_int, y_pt_int), self.hp_radius)

        # Draw center point of tennis court
        x_ct, y_ct = line_intersection((kps[0][0], kps[0][1], kps[3][0], kps[3][1]), (kps[1][0], kps[1][1], kps[2][0], kps[2][1]))
        draw_gaussian(hm_hp[self.num_joints], (int(x_ct/self.scale), int(y_ct/self.scale)), self.hp_radius)
        
        return inp, hm_hp, np.array(kps, dtype=int), img_name[:-4]

    def __len__(self):
        return len(self.data)
