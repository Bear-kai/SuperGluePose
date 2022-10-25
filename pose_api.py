from pathlib import Path
import argparse
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import open3d as o3d
import time
from models.matching import Matching
from models.utils import scale_intrinsics, estimate_pose, make_matching_plot_fast, read_image

torch.set_grad_enabled(False)


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--resize', type=int, nargs='+', default=[-1],              # w,h = [640, 480]   
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', type=bool, default=False,
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,          # 0.002
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--opencv_display', type=bool, default=False, help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--show_keypoints', type=bool, default=False, help='Plot the keypoints in addition to the matches')
    
    opt = parser.parse_args()

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    # print(opt)
    return opt


def get_config(opt):
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    return config


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)     # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(curr_pose_pred, curr_pose_gt):
    euc_t = np.linalg.norm(curr_pose_pred['t'] - curr_pose_gt['t'])
    error_t = angle_error_vec(curr_pose_pred['t'], curr_pose_gt['t'])
    error_t = np.minimum(error_t, 180 - error_t)    # ambiguity of E estimation
    error_R = angle_error_mat(curr_pose_pred['R'], curr_pose_gt['R'])
    
    print('t_gt  ', curr_pose_gt['t'])
    print('t_pred', curr_pose_pred['t'])
    print('R_gt  ----------')
    print(curr_pose_gt['R'])
    print('R_pred----------')
    print(curr_pose_pred['R'])
    print('euc_t = ', euc_t)     # error in norm
    print('err_t = ', error_t)   # error in angle
    print('err_R = ', error_R)

    return error_t, error_R


def get_text(kpts0, kpts1, mkpts0, stem0, stem1, matching):
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]
    
    # Display extra parameter info.
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {}:{}'.format(stem0, stem1),
    ]

    return text, small_text


def get_filter_idx(mkpts0, kps_filter):
    tmp_dict = {}
    for i in range(mkpts0.shape[0]):
        tmp_dict[tuple(list(mkpts0[i]))] = i
        
    ind_ls = []
    for i in range(kps_filter.shape[0]):
        ind_ls.append(tmp_dict[tuple(list(kps_filter[i]))])
    
    return np.array(ind_ls)


def get_seg_mask(mask_path):
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  
    img[img>0] = 1
    seg_mask = img.astype(bool)

    return seg_mask


def get_depth_ycb(depth_path, resize, scale): 
    depth_img = cv2.imread(str(depth_path), -1)
    depth_scale=0.1                                         # for YCB-V data
    depth_img = np.float32(depth_img * depth_scale / 1000)  # depth value in meters!
    if scale != (1.0, 1.0):                                 
        depth_img = cv2.resize(depth_img, resize)           # resize = (w, h) 

    return depth_img


def backproj(uv_kps, depth, camK, scale=1):
    """ 
    Params: 
        uv_kps: nx2 array, each row is (x,y);
        depth: hxw array, depth image; if depth is not scaled in meters, please set the param `scale`!
        camK: 3x3 array, the camera intrinsics;
    Output:
        pts: the point cloud
        z_mask: point mask, in case there is no valid depth value.
    """ 
    uv_kps = uv_kps.astype(int)
    intrinsics_inv = np.linalg.inv(camK) 
    ones = np.ones([uv_kps.shape[0], 1])
    uv_grid = np.concatenate((uv_kps, ones), axis=1)    # [num, 3]
    xyz = uv_grid @ intrinsics_inv.T                    # [num, 3]

    z = depth[uv_kps[:,1], uv_kps[:,0]].astype(np.float32)
    z_mask = (z > 0)
    pts = xyz * z[:, np.newaxis]  # / xyz[:, -1:]
    
    return pts * scale, z_mask


def estimate_pose_ransac(pcd0, pcd1):
    """ matched pcd0/pcd1: nx3 array. """

    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd0))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd1))
    corres_arr = np.stack([np.arange(pcd0.shape[0]), np.arange(pcd0.shape[0])], axis=1)
    corres = o3d.utility.Vector2iVector(corres_arr)
    
    max_corres_dist = 0.02                           # if dist(p1,p2) > max_corres_dist, then pair (p1,p2) are regarded as outlier. This threshold has effect on `fitness`. A big value means fitness=1.
    # estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    # criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=1000, confidence=0.999)
    
    start = time.time()
    reg = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source, target, corres, max_corres_dist,
            # estimation_method, ransac_n=3, criteria=criteria
    )
    print('\ntime elapses: ', time.time() - start)
    print('fitness: ', reg.fitness)                  # The overlapping area (# of inlier correspondences / # of points in source). Higher is better.
    print('inlier_rmse: ', reg.inlier_rmse, '\n')    # RMSE of all inlier correspondences. Lower is better.
    # print(reg.transformation)

    ret = (reg.transformation[:3,:3], reg.transformation[:3,3], reg.fitness)

    return ret
    

class PoseAPI():
    def __init__(self, opt, device, camK0, camK1) -> None:
        self.opt = opt
        self.camK0 = camK0      # for frame0
        self.camK1 = camK1      # for frame1
        config = get_config(self.opt)
        self.matching = Matching(config).eval().to(device)
    
    def forward(self, inp0, inp1, scales0, scales1, last_pose_ls, seg_mask_ls, depth0, depth1):  
        """  
        Params:
            inp0/inp1: tensor of gray image_0/image_1;
            scales0/scales1: array, scale factor for resizing image; if not resize, set scale=1.0
            last_pose_ls: list of {'R':array, 't':array}, pose from object coord to image_0 coord.
            seg_mask_ls: list of bool array
            depth0/depth1: array, the input depth image need to be scaled in meters! 
        Output:
            current_pose_ls: list of {'R':array, 't':array}, pose from object coord to image_1 coord.
            results_extra_ls: list of {'mkpts0':array,...}, only used for visualization.
        """
        # Perform the matching.
        pred = self.matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # Scale the intrinsics to resized image.
        K0 = scale_intrinsics(self.camK0, scales0)
        K1 = scale_intrinsics(self.camK1, scales1)

        # build kps_mask 
        kps_mask = np.zeros((self.opt.resize[1], self.opt.resize[0]), dtype=bool)
        index = mkpts0.astype(int)
        kps_mask[index[:,1], index[:,0]] = 1

        # Look through seg_mask list to estimate pose
        current_pose_ls, results_extra_ls = [], []
        for i, seg_mask in enumerate(seg_mask_ls):
            comb_mask = kps_mask & seg_mask
            inds = np.nonzero(comb_mask)
            mkpts0_filter = np.stack([inds[1], inds[0]], axis=1).astype(np.float32)
            filter_ind = get_filter_idx(mkpts0, mkpts0_filter)
            mkpts1_filter = mkpts1[filter_ind]
            mconf_filter = mconf[filter_ind]

            # backproject to get point cloud
            mkpts0_pcd, z_mask0 = backproj(mkpts0_filter, depth0, K0)
            mkpts1_pcd, z_mask1 = backproj(mkpts1_filter, depth1, K1)
            z_mask = z_mask0 & z_mask1

            # Estimate camera delta pose.
            ret = estimate_pose_ransac(mkpts0_pcd[z_mask], mkpts1_pcd[z_mask])
            if ret:
                R, t, inliers = ret
            else:
                R, t = np.eye(3), np.zeros(3)

            # Estimate current object pose
            pred_R = np.dot(R, last_pose_ls[i]['R'])
            pred_t = np.dot(R, last_pose_ls[i]['t']) + t
            current_pose = {'R':pred_R, 't':pred_t}
            current_pose_ls.append(current_pose)

            results_extra = {'mconf':mconf_filter[z_mask], 'mkpts0':mkpts0_filter[z_mask], 'mkpts1':mkpts1_filter[z_mask], 
                            'kpts0':kpts0, 'kpts1':kpts1, }
            results_extra_ls.append(results_extra)

        return current_pose_ls, results_extra_ls

    def forward_epipolar(self, inp0, inp1, scales0, scales1, last_pose_ls, seg_mask_ls):
        """ 
        Params:
            inp0/inp1: tensor of gray image_0/image_1;
            scales0/scales1: array, scale factor for resizing image;
            last_pose_ls: list of {'R':array, 't':array}, pose from object coord to image_0 coord.
            seg_mask_ls: list of bool array
        Output:
            current_pose_ls: list of {'R':array, 't':array}, pose from object coord to image_1 coord.
            results_extra_ls: list of {'mkpts0':array,...}, only used for visualization.
        """
        # Perform the matching.
        pred = self.matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # Scale the intrinsics to resized image.
        K0 = scale_intrinsics(self.camK0, scales0)
        K1 = scale_intrinsics(self.camK1, scales1)

        # build kps_mask 
        kps_mask = np.zeros((self.opt.resize[1], self.opt.resize[0]), dtype=bool)
        index = mkpts0.astype(int)
        kps_mask[index[:,1], index[:,0]] = 1

        # Look through seg_mask list to estimate pose
        current_pose_ls, results_extra_ls = [], []
        for i, seg_mask in enumerate(seg_mask_ls):
            comb_mask = kps_mask & seg_mask
            inds = np.nonzero(comb_mask)
            mkpts0_filter = np.stack([inds[1], inds[0]], axis=1).astype(np.float32)
            filter_ind = get_filter_idx(mkpts0, mkpts0_filter)
            mkpts1_filter = mkpts1[filter_ind]
            mconf_filter = mconf[filter_ind]

            # Estimate camera delta pose.
            thresh = 1.    # In pixels relative to resized image size.
            ret = estimate_pose(mkpts0_filter, mkpts1_filter, K0, K1, thresh)
            if ret:
                R, t, inliers = ret
            else:
                R, t = np.eye(3), np.zeros(3)

            # Estimate current object pose
            pred_R = np.dot(R, last_pose_ls[i]['R'])
            pred_t = np.dot(R, last_pose_ls[i]['t']) + t
            current_pose = {'R':pred_R, 't':pred_t}
            current_pose_ls.append(current_pose)

            results_extra = {'mconf':mconf_filter, 'mkpts0':mkpts0_filter, 'mkpts1':mkpts1_filter, 
                            'kpts0':kpts0, 'kpts1':kpts1, }
            results_extra_ls.append(results_extra)

        return current_pose_ls, results_extra_ls
        

def example():

    # sample params
    vis_and_save = True  # False
    in_dir = 'my_test_data/'
    out_dir = 'my_test_result/'
    name0, name1 = ['rgb/000001.jpg', 'rgb/000018.jpg']         # two images from YCB-V  # ['000002/000055.png', '000002/000056.png']  # real-sense data
    mask_path = 'mask/000001.png'
    depth0_path, depth1_path = ['depth/000001.png', 'depth/000018.png']
    gt_pose_0 = 'gt_pose_objTOimg1.npz'                         # 对应YCB-V 000001.jpg
    gt_pose_1 = 'gt_pose_objTOimg18.npz'                        # 对应YCB-V 000018.jpg

    camK = np.array([1066.778, 0.0, 312.9869079589844,          # for YCB-V
                     0.0, 1067.487, 241.3108977675438, 
                     0.0, 0.0, 1.0]).reshape((3,3)) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # init
    opt = arg_parse()
    api = PoseAPI(opt, device, camK, camK)

    # Load the image pair
    input_dir = Path(in_dir)
    if vis_and_save:
        output_dir = Path(out_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    stem0, stem1 = Path(name0).stem, Path(name1).stem
    image0, inp0, scales0 = read_image(input_dir / name0, device, opt.resize, 0, opt.resize_float)
    image1, inp1, scales1 = read_image(input_dir / name1, device, opt.resize, 0, opt.resize_float)

    # load seg_mask and depth image
    if len(opt.resize) == 1:                 
        opt.resize = image0.shape[::-1]
    # seg_mask = np.ones((opt.resize[1], opt.resize[0]), dtype=bool)    # fake mask
    seg_mask = get_seg_mask(input_dir / mask_path)

    depth0 = get_depth_ycb(input_dir / depth0_path, opt.resize, scales0)
    depth1 = get_depth_ycb(input_dir / depth1_path, opt.resize, scales1)

    # Load pose ground truth -- only for testing
    last_pose_gt = np.load(str(input_dir / gt_pose_0))
    curr_pose_gt = np.load(str(input_dir / gt_pose_1))

    # estimate pose: the returned `current_pose_pred` is what we want !
    # curr_pose_pred, results_extra = api.forward_epipolar(inp0, inp1, scales0, scales1, [last_pose_gt], [seg_mask])        # 基于对极几何，无需深度图，但存在尺度问题
    curr_pose_pred, results_extra = api.forward(inp0, inp1, scales0, scales1, [last_pose_gt], [seg_mask], depth0, depth1)   # 基于3D-3D的ransac匹配，需要提供深度图
    
    # only for testing, since we have no `curr_pose_gt` in practice.
    err_t, err_R = compute_pose_error(curr_pose_pred[0], curr_pose_gt)
    
    # Visualize the matches.
    if vis_and_save:
        results_extra = results_extra[0]
        color = cm.jet(results_extra['mconf'])  # eg. (#keypts=38, 4)
        text, small_text = get_text(results_extra['kpts0'], results_extra['kpts1'], 
                                    results_extra['mkpts0'], stem0, stem1, api.matching)
        viz_path = output_dir / '{}_{}_matches.png'.format(stem0, stem1)
        make_matching_plot_fast(image0, image1, results_extra['kpts0'], results_extra['kpts1'], 
                                results_extra['mkpts0'], results_extra['mkpts1'],
                                color, text, viz_path, opt.show_keypoints, 10,
                                opt.opencv_display, 'Matches', small_text)


if __name__ == '__main__':
    example()
        



# 基于open3d的registration ransac，预测结果具有随机性；

# ------------- 3D-3D匹配 --------------
# time elapses:  0.0009970664978027344
# fitness:  0.8947368421052632
# inlier_rmse:  0.00773584319842859

# t_gt   [0.04498924 0.21220969 1.1689751 ]
# t_pred [0.04140051 0.21395826 1.16573447]
# R_gt  ----------
# [[ 0.11900599  0.81035012 -0.57373381]
#  [-0.98100132  0.00679719 -0.19388251]
#  [-0.15321286  0.58590674  0.79576325]]
# R_pred----------
# [[ 0.11022404  0.83827535 -0.53399021]
#  [-0.98196003  0.00879514 -0.18888517]
#  [-0.15364124  0.54517671  0.82412189]]
# euc_t =  0.00514180557261773  欧式距离
# err_t =  0.2014050584976629   角度差异
# err_R =  2.8444207257631398   角度差异

# ------------- 2D-2D对极几何 ----------------
# t_gt   [0.04498924 0.21220969 1.1689751 ]
# t_pred [-0.04546463  0.23452019  1.99105722]
# R_gt:
# [[ 0.11900599  0.81035012 -0.57373381]
#  [-0.98100132  0.00679719 -0.19388251]
#  [-0.15321286  0.58590674  0.79576325]]
# R_pred:
# [[ 0.13998379  0.97221866 -0.18760589]
#  [-0.99001734  0.13428087 -0.04283461]
#  [-0.0164527   0.1917292   0.98130998]]
# err_t =  4.977471265193023  角度差异
# err_R =  26.60023506627283  角度差异