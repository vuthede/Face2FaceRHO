import argparse
from options.parse_config import Face2FaceRHOConfigParse
from models import create_model
import os
import torch
import numpy as np
from util.landmark_image_generation import LandmarkImageGeneration
from util.util import (
    read_target,
    load_coeffs,
    load_landmarks
)
import cv2
from util.util import tensor2im
from external.decalib.utils.rotation_converter import batch_rodrigues

def parse_args():
    """Configurations."""
    parser = argparse.ArgumentParser(description='test process of Face2FaceRHO')
    parser.add_argument('--config', type=str, required=True,
                        help='.ini config file name')
    parser.add_argument('--src_img', type=str, required=True,
                        help='input source actor image (.jpg, .jpg, .jpeg, .png)')
    parser.add_argument('--src_headpose', type=str, required=True,
                        help='input head pose coefficients of source image (.txt)')
    parser.add_argument('--src_landmark', type=str, required=True,
                        help='input facial landmarks of source image (.txt)')
    parser.add_argument('--drv_headpose', type=str, required=True,
                        help='input head pose coefficients of driving image (.txt)')
    parser.add_argument('--drv_landmark', type=str, required=True,
                        help='input driving facial landmarks (.txt, reconstructed by using shape coefficients '
                             'of the source actor and expression and head pose coefficients of the driving actor)')

    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'test_case', 'result'),
                        help='output directory')

    return _check_args(parser.parse_args())


def _check_args(args):
    if args is None:
        raise RuntimeError('Invalid arguments!')
    return args


def load_data(opt, headpose_file, landmark_file, img_file=None, load_img=True):
    face = dict()
    if load_img:
        face['img'] = read_target(img_file, opt.output_size)
    face['headpose'] = torch.from_numpy(np.array(load_coeffs(headpose_file))).float()
    face['landmarks'] = torch.from_numpy(np.array(load_landmarks(landmark_file))).float()
    return face

def draw_landmarks(img, lmks, color=(255,0,0)):
    img = np.ascontiguousarray(img)
    for a in lmks:
        cv2.circle(img,(int(round(a[0])), int(round(a[1]))), 1, color, -1, lineType=cv2.LINE_AA)

    return img

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

if __name__ == '__main__':
    args = parse_args()
    config_parse = Face2FaceRHOConfigParse()
    opt = config_parse.get_opt_from_ini(args.config)
    config_parse.setup_environment()

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    src_face = load_data(opt, args.src_headpose, args.src_landmark, args.src_img)
    # drv_face = load_data(opt, args.src_headpose, args.src_landmark, load_img=False)

    drv_face = load_data(opt, args.drv_headpose, args.drv_landmark, load_img=False)

    landmark_img_generator = LandmarkImageGeneration(opt)

    # off-line stage
    src_face['landmark_img'] = landmark_img_generator.generate_landmark_img(src_face['landmarks'])
    src_face['landmark_img'] = [value.unsqueeze(0) for value in src_face['landmark_img']]
    model.set_source_face(src_face['img'].unsqueeze(0), src_face['headpose'].unsqueeze(0))

    # on-line stage
    drv_face['landmark_img'] = landmark_img_generator.generate_landmark_img(drv_face['landmarks'])
    drv_face['landmark_img'] = [value.unsqueeze(0) for value in drv_face['landmark_img']]

    # rotation_m = batch_rodrigues(drv_face['headpose'].unsqueeze(0))
    # import pdb; pdb.set_trace();
    def synthesize_pose(pose,  delta_pitch=0, delta_yaw=30, delta_roll=0):
        from scipy.spatial.transform import Rotation
        rotation_axis = drv_face['headpose'].detach().cpu().numpy()
        euler = Rotation.from_rotvec(rotation_axis[:3]).as_euler('xyz', degrees=True)
        # euler = Rotation.from_rotvec(rotation_axis[3:]).as_euler('xyz', degrees=True)

        euler += np.array([delta_pitch, delta_yaw, delta_roll])
        rotation_axis_synthesized = Rotation.from_euler('xyz', euler, degrees=True).as_rotvec()

        rotation_axis_synthesized = torch.FloatTensor([rotation_axis_synthesized[0],rotation_axis_synthesized[1], rotation_axis_synthesized[2],\
                        rotation_axis[3], rotation_axis[4], rotation_axis[5]])
        # rotation_axis_synthesized = torch.FloatTensor([rotation_axis[0],rotation_axis[1], rotation_axis[2],\
        #                 rotation_axis_synthesized[0], rotation_axis_synthesized[1], rotation_axis_synthesized[2]])
        
        return rotation_axis_synthesized

    # drv_face['headpose']= synthesize_pose(drv_face['headpose'], delta_pitch=0, delta_yaw=2, delta_roll=0)

    # import pdb; pdb.set_trace()
    # cv2.imshow("a",src_face['landmark_img'])
    # cv2.imshow("b",drv_face['landmark_img'])

    # if cv2.waitKey(0)==27:
    #     cv2.destroyAllWindows()
    print(drv_face['headpose'], src_face['headpose'])
    model.reenactment(src_face['landmark_img'], drv_face['headpose'].unsqueeze(0), drv_face['landmark_img'])

    plot_lmks_frame(src_face['landmarks'], drv_face['landmarks'])

    visual_results = model.get_current_visuals()
    output_file_name = args.output_dir + "/result.png"
    im = tensor2im(visual_results['fake'])
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file_name, im)

    cv2.waitKey(0)