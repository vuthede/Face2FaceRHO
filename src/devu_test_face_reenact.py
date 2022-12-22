import argparse
import torch
import json
import os
import copy
import numpy as np
from external.decalib.utils.config import cfg as deca_cfg
from external.decalib.deca import DECA
from external.decalib.datasets import datasets
from external.decalib.models.FLAME import FLAME
from util.util import (
    save_coeffs,
    save_landmarks
)

import cv2

from options.parse_config import Face2FaceRHOConfigParse
from models import create_model
from util.landmark_image_generation import LandmarkImageGeneration
from util.util import (
    read_target,
    load_coeffs,
    load_landmarks
)
from util.util import tensor2im


device = 'cpu'

class FLAMEFitting:
    def __init__(self):
        self.deca = DECA(config=deca_cfg, device=device)

    def fitting(self, img_name):
        testdata = datasets.TestData(img_name, iscrop=True,face_detector='fan', sample_step=10)
        input_data = testdata[0]
        images = input_data['image'].to(device)[None, ...]
        with torch.no_grad():
            codedict = self.deca.encode(images)
            codedict['tform'] = input_data['tform'][None, ...]
            original_image = input_data['original_image'][None, ...]
            _, _, h, w = original_image.shape
            params = self.deca.ensemble_3DMM_params(codedict, image_size=deca_cfg.dataset.image_size, original_image_size=h)
        return params


class PoseLandmarkExtractor:
    def __init__(self):
        self.flame = FLAME(deca_cfg.model)

        with open(os.path.join(deca_cfg.deca_dir, 'data', 'pose_transform_config.json'), 'r') as f:
            pose_transform = json.load(f)

        self.scale_transform = pose_transform['scale_transform']
        self.tx_transform = pose_transform['tx_transform']
        self.ty_transform = pose_transform['ty_transform']
        self.tx_scale = 0.256 # 512 / 2000
        self.ty_scale = - self.tx_scale

    @staticmethod
    def transform_points(points, scale, tx, ty):
        trans_matrix = torch.zeros((1, 4, 4), dtype=torch.float32)
        trans_matrix[:, 0, 0] = scale
        trans_matrix[:, 1, 1] = -scale
        trans_matrix[:, 2, 2] = 1
        trans_matrix[:, 0, 3] = tx
        trans_matrix[:, 1, 3] = ty
        trans_matrix[:, 3, 3] = 1

        batch_size, n_points, _ = points.shape
        points_homo = torch.cat([points, torch.ones([batch_size, n_points, 1], dtype=points.dtype)], dim=2)
        points_homo = points_homo.transpose(1, 2)
        trans_points = torch.bmm(trans_matrix, points_homo).transpose(1, 2)
        trans_points = trans_points[:, :, 0:3]
        return trans_points

    def get_project_points(self, shape_params, expression_params, pose, scale, tx, ty):
        shape_params = torch.tensor(shape_params).unsqueeze(0)
        expression_params = torch.tensor(expression_params).unsqueeze(0)
        pose = torch.tensor(pose).unsqueeze(0)
        verts, landmarks3d = self.flame(
            shape_params=shape_params, expression_params=expression_params, pose_params=pose)
        trans_landmarks3d = self.transform_points(landmarks3d, scale, tx, ty)
        trans_landmarks3d = trans_landmarks3d.squeeze(0).cpu().numpy()
        return trans_landmarks3d[:, 0:2].tolist(), trans_landmarks3d

    def calculate_nose_tip_tx_ty(self, shape_params, expression_params, pose, scale, tx, ty):
        front_pose = copy.deepcopy(pose)
        front_pose[0] = front_pose[1] = front_pose[2] = 0
        front_landmarks3d, _ = self.get_project_points(shape_params, expression_params, front_pose, scale, tx, ty)
        original_landmark3d,_ = self.get_project_points(shape_params, expression_params, pose, scale, tx, ty)
        nose_tx = original_landmark3d[30][0] - front_landmarks3d[30][0]
        nose_ty = original_landmark3d[30][1] - front_landmarks3d[30][1]
        return nose_tx, nose_ty

    def get_pose(self, shape_params, expression_params, pose, scale, tx, ty):
        nose_tx, nose_ty = self.calculate_nose_tip_tx_ty(
            shape_params, expression_params, pose, scale, tx, ty)
        transformed_axis_angle = [
            float(pose[0]),
            float(pose[1]),
            float(pose[2])
        ]
        transformed_tx = self.tx_transform + self.tx_scale * (tx + nose_tx)
        transformed_ty = self.ty_transform + self.ty_scale * (ty + nose_ty)
        transformed_scale = scale / self.scale_transform
        return transformed_axis_angle + [transformed_tx, transformed_ty, transformed_scale]


def load_model_reenact():
    config_parse = Face2FaceRHOConfigParse()
    opt = config_parse.get_opt_from_ini("./src/config/test_face2facerho.ini")
    config_parse.setup_environment()
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    landmark_img_generator = LandmarkImageGeneration(opt)

    return model, landmark_img_generator

def draw_landmarks(img, lmks, color=(255,0,0)):
    img = np.ascontiguousarray(img)
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 1, color, -1, lineType=cv2.LINE_AA)

    return img


def lmks2box(lmks, img, expand=1.0):
    xy = np.min(lmks, axis=0).astype(np.int32) 
    zz = np.max(lmks, axis=0).astype(np.int32)

    xy[1] = max(xy[1], 0) 
    wh = zz - xy + 1

    center = (xy + wh/2).astype(np.int32)
    # EXPAND=1.1
    EXPAND=expand
    boxsize = int(np.max(wh)*EXPAND)
    xy = center - boxsize//2
    x1, y1 = xy
    x2, y2 = xy + boxsize
    height, width, _ = img.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    return [x1, y1, x2, y2]

def load_lmks2d(txt):
    with open(txt, 'r') as f:
        lmks = f.readline().split(",")
        lmks = [float(l) for l in lmks]
        lmks = np.reshape(lmks, (-1, 2))
        
        return np.array(lmks)

def squarebox(img, box, expand=1.0):
    ori_shape = img.shape
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)//2, (y1+y2)//2
    w = max(x2-x1, y2-y1)*expand
    x1 = cx - w//2
    y1 = cy - w//2
    x2 = cx + w//2
    y2 = cy + w//2

    x1 = int(max(x1, 0))
    y1 = int(max(y1+(y2-y1)*0, 0))
    x2 = int(min(x2-(x1-x1)*0, ori_shape[1]-1))
    y2 = int(min(y2, ori_shape[0]-1))

    return x1, y1, x2, y2

def crop_target_image():
    from ibug.face_detection import RetinaFacePredictor
    face_detector = RetinaFacePredictor(threshold=0.8, device='cuda', model=RetinaFacePredictor.get_model())

    # fullimg = "/home/vuthede/Pictures/depose.jpg"
    fullimg = "/media/vuthede/Backup Plus/AI/VinAI/datasets/biwi/faces_0/01/frame_00139_rgb.png"
    frame = cv2.imread(fullimg)
    faces = face_detector(frame, rgb=False)
    box = faces[0][:4]
    x1, y1, x2, y2 = squarebox(frame, box, expand=2.2)
    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (512, 512))
    cv2.imwrite("target.png", face)


if __name__ == '__main__':

    # 3DMM fitting by DECA: Detailed Expression Capture and Animation using FLAME model
    face_fitting = FLAMEFitting()
    src_img = "./test_case/source/source.jpg"
    src_params = face_fitting.fitting(src_img)
    # Crop target
    # crop_target_image()
    # drv_params = face_fitting.fitting("target.png")


  

    print(f'Load fitting model')
    model, landmark_img_generator = load_model_reenact()
    print(f'Loaded reenact model')

    # import pdb; pdb.set_trace()
    # calculate head pose and facial landmarks for the source and driving face images
    pose_lml_extractor = PoseLandmarkExtractor()
    src_headpose = pose_lml_extractor.get_pose(
        src_params['shape'], src_params['exp'], src_params['pose'],
        src_params['scale'], src_params['tx'], src_params['ty'])
    src_lmks, src_lmks_3d = pose_lml_extractor.get_project_points(
        src_params['shape'], src_params['exp'], src_params['pose'],
        src_params['scale'], src_params['tx'], src_params['ty'])

    # Note that the driving head pose and facial landmarks are calculated using the shape parameters of the source image
    # in order to eliminate the interference of the driving actor's identity.
    ################## Synthesize pose
    def synthesize_pose(pose,  delta_pitch=0, delta_yaw=30, delta_roll=0):
        from scipy.spatial.transform import Rotation
        rotation_axis = pose
        euler = Rotation.from_rotvec(rotation_axis[:3]).as_euler('xyz', degrees=True)
        euler += np.array([delta_pitch, delta_yaw, delta_roll])
        rotation_axis_synthesized_face = Rotation.from_euler('xyz', euler, degrees=True).as_rotvec()

        # euler_yaw = Rotation.from_rotvec(rotation_axis[3:]).as_euler('xyz', degrees=True)
        # euler_yaw += np.array([delta_pitch, delta_yaw, delta_roll])
        # rotation_axis_synthesized_yaw = Rotation.from_euler('xyz', euler_yaw, degrees=True).as_rotvec()

        rotation_axis_synthesized = torch.FloatTensor([rotation_axis_synthesized_face[0],rotation_axis_synthesized_face[1], rotation_axis_synthesized_face[2],\
                        rotation_axis[3], rotation_axis[4], rotation_axis[5]])
        
        return rotation_axis_synthesized

    def synthesize_R(delta_pitch=0, delta_yaw=30, delta_roll=0):
        from scipy.spatial.transform import Rotation
        
        Rot = Rotation.from_euler('xyz', [delta_pitch, delta_yaw, delta_roll], degrees=True).as_matrix()

        return Rot

    delta_yaw = 0
    delta_pitch = 0
    delta_roll = 0

    
    # def plot_lmks_frame(src_lmks, drv_lmks):
    #     src_lmks = np.array(src_lmks)
    #     drv_lmks = np.array(drv_lmks)
    #     src_lmks = (src_lmks+1)/2 * 512
    #     drv_lmks = (drv_lmks+1)/2 * 512

    #     src_zero = np.zeros((512,512,3))
    #     drv_zero = np.zeros((512,512,3))
    #     src_zero = draw_landmarks(src_zero, src_lmks)
    #     drv_zero = draw_landmarks(drv_zero, drv_lmks)
    #     concat_img = np.hstack([src_zero, drv_zero])

    #     cv2.imshow("lmks", concat_img)

    def plot_lmks_frame(src_lmks, drv_lmks):
        src_lmks = np.array(src_lmks)
        drv_lmks = np.array(drv_lmks)
        src_lmks = (src_lmks+1)/2 * 512
        drv_lmks = (drv_lmks+1)/2 * 512

        src_zero = np.zeros((512,512,3))
        drv_zero = np.zeros((512,512,3))
        src_zero = draw_landmarks(src_zero, src_lmks)
        drv_zero = draw_landmarks(src_zero, drv_lmks, color=(0,255,255))
        # concat_img = np.hstack([src_zero, drv_zero])

        cv2.imshow("lmks", drv_zero)


    while True:
        print(f'src_params_posr :{src_params["pose"]}')
        syn_param_pose = synthesize_pose(src_params['pose'], delta_pitch=delta_pitch, delta_yaw=delta_yaw, delta_roll=delta_roll)
        syn_param_pose[3:] = torch.FloatTensor(src_params['pose'])[3:]

        drv_headpose = pose_lml_extractor.get_pose(
            src_params['shape'], src_params['exp'], syn_param_pose,
            src_params['scale'], src_params['tx'], src_params['ty'])
        drv_lmks, drv_lmks3d = pose_lml_extractor.get_project_points(
            src_params['shape'], src_params['exp'], syn_param_pose,
            src_params['scale'], src_params['tx'], src_params['ty'])
        drv_lmks = np.array(drv_lmks)

        # drv_params['pose'][3:] = src_params['pose'][3:]
        # drv_headpose = pose_lml_extractor.get_pose(
        #     src_params['shape'], src_params['exp'], drv_params['pose'],
        #     drv_params['scale'], drv_params['tx'], drv_params['ty'])
        # drv_lmks, drv_lmks_3d = pose_lml_extractor.get_project_points(
        #     src_params['shape'], src_params['exp'], drv_params['pose'],
        #     drv_params['scale'], drv_params['tx'], drv_params['ty'])


        # translate = np.mean(src_lmks, axis=0) - np.mean(drv_lmks, axis=0) + np.array([0.2,0.0])
        # drv_lmks += translate

        # drv_headpose = synthesize_pose(src_headpose, delta_pitch=delta_pitch, delta_yaw=delta_yaw, delta_roll=0)
        # R_2_target = synthesize_R(delta_pitch=delta_pitch, delta_yaw=delta_yaw, delta_roll=0)
        # drv_lmks_3d = (R_2_target@(src_lmks_3d-src_lmks_3d[30]).T).T + src_lmks_3d[30]
        # drv_lmks = drv_lmks_3d[:, 0:2].tolist()
        # import pdb;pdb.set_trace()
        plot_lmks_frame(src_lmks, drv_lmks)


        # drv_headpose = src_headpose
        # drv_lmks = src_lmks
        
        # Reenact
        config_parse = Face2FaceRHOConfigParse()
        opt = config_parse.get_opt_from_ini("./src/config/test_face2facerho.ini")
        src_face_norm = read_target(src_img, opt.output_size) 

        src_pose_tensor = torch.FloatTensor(src_headpose)
        model.set_source_face(src_face_norm.unsqueeze(0),src_pose_tensor.unsqueeze(0))

        src_face_landmark_img = landmark_img_generator.generate_landmark_img(torch.FloatTensor((src_lmks)))
        src_face_landmark_img = [value.unsqueeze(0) for value in src_face_landmark_img]

        drv_face_landmark_img = landmark_img_generator.generate_landmark_img(torch.FloatTensor(drv_lmks))
        drv_face_landmark_img = [value.unsqueeze(0) for value in drv_face_landmark_img]

        syn_pose_tensor = torch.FloatTensor(drv_headpose)
        model.reenactment(src_face_landmark_img, syn_pose_tensor.unsqueeze(0), drv_face_landmark_img)

        visual_results = model.get_current_visuals()
        im = tensor2im(visual_results['fake'])
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        im_origin_src = cv2.imread(src_img)
        concat_img = np.hstack([im_origin_src, im])

        cv2.imshow("Image", concat_img)

        k = cv2.waitKey(0)

        if k ==27:
            break
        
        ### Eye
        unit_angle = 5
        if ord('a')==k:
            delta_yaw = delta_yaw - unit_angle
        if ord('d')==k:
            delta_yaw = delta_yaw + unit_angle
        
        if ord('s')==k:
            delta_pitch = delta_pitch - unit_angle
        if ord('w')==k:
            delta_pitch = delta_pitch + unit_angle
        
        if ord('j')==k:
            delta_roll = delta_roll - unit_angle
        if ord('l')==k:
            delta_roll = delta_roll + unit_angle
    
    cv2.destroyAllWindows()

