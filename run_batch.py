# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from bundlesdf import *
import argparse
import os, sys, glob
import traceback  # Added for detailed error logging
import copy # Added missing import usually required for deepcopy
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from segmentation_utils import Segmenter
from scipy.spatial.transform import Rotation as R


def run_one_video(video_dir, out_folder, use_segmenter=False, use_gui=False, stride=1, debug_level=2):
    """
    Runs BundleSDF tracking on a single video sequence.
    """
    print(f"--- Processing: {video_dir} ---")
    print(f"--- Outputting to: {out_folder} ---")
    
    set_seed(0)

    # Clean and create output directory
    os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}')

    cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml", 'r'))
    cfg_bundletrack['SPDLOG'] = int(debug_level)
    cfg_bundletrack['depth_processing']["percentile"] = 95
    cfg_bundletrack['erode_mask'] = 3
    cfg_bundletrack['debug_dir'] = out_folder + '/'
    cfg_bundletrack['bundle']['max_BA_frames'] = 10
    cfg_bundletrack['bundle']['max_optimized_feature_loss'] = 0.03
    cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.02
    cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 30
    cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.01
    cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 20
    cfg_bundletrack['feature_corres']['map_points'] = True
    cfg_bundletrack['feature_corres']['resize'] = 400
    cfg_bundletrack['feature_corres']['rematch_after_nerf'] = True
    cfg_bundletrack['keyframe']['min_rot'] = 5
    cfg_bundletrack['ransac']['inlier_dist'] = 0.01
    cfg_bundletrack['ransac']['inlier_normal_angle'] = 20
    cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.02
    cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 30
    cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.01
    cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 10
    cfg_bundletrack['p2p']['max_dist'] = 0.02
    cfg_bundletrack['p2p']['max_normal_angle'] = 45
    cfg_track_dir = f'{out_folder}/config_bundletrack.yml'
    yaml.dump(cfg_bundletrack, open(cfg_track_dir, 'w'))

    cfg_nerf = yaml.load(open(f"{code_dir}/config.yml", 'r'))
    cfg_nerf['continual'] = True
    cfg_nerf['trunc_start'] = 0.01
    cfg_nerf['trunc'] = 0.01
    cfg_nerf['mesh_resolution'] = 0.005
    cfg_nerf['down_scale_ratio'] = 1
    cfg_nerf['fs_sdf'] = 0.1
    cfg_nerf['far'] = cfg_bundletrack['depth_processing']["zfar"]
    cfg_nerf['datadir'] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
    cfg_nerf['notes'] = ''
    cfg_nerf['expname'] = 'nerf_with_bundletrack_online'
    cfg_nerf['save_dir'] = cfg_nerf['datadir']
    cfg_nerf_dir = f'{out_folder}/config_nerf.yml'
    yaml.dump(cfg_nerf, open(cfg_nerf_dir, 'w'))

    if use_segmenter:
        segmenter = Segmenter()

    tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5, use_gui=use_gui)

    reader = YcbineoatReader(video_dir=video_dir, shorter_side=360)

    for i in range(0, len(reader.color_files), stride):
        color_file = reader.color_files[i]
        color = cv2.imread(color_file)
        H0, W0 = color.shape[:2]
        depth = reader.get_depth(i)
        H, W = depth.shape[:2]
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        if i == 0:
            mask = reader.get_mask(0)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            if use_segmenter:
                mask = segmenter.run(color_file.replace('rgb', 'masks'))
        else:
            if use_segmenter:
                mask = segmenter.run(color_file.replace('rgb', 'masks'))
            else:
                mask = reader.get_mask(i)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        if cfg_bundletrack['erode_mask'] > 0:
            kernel = np.ones((cfg_bundletrack['erode_mask'], cfg_bundletrack['erode_mask']), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel)

        id_str = reader.id_strs[i]
        pose_in_model = np.eye(4)

        K = reader.K.copy()

        tracker.run(color, depth, K, id_str, mask=mask, occ_mask=None, pose_in_model=pose_in_model)

    tracker.on_finish()

    # Pass specific video_dir to global nerf, not the root dir from args
    run_one_video_global_nerf(video_dir=video_dir, out_folder=out_folder)


def run_one_video_global_nerf(video_dir, out_folder):
    """
    Runs the global NeRF refinement step.
    IMPORTANT: Now accepts video_dir explicitly so it works in batch mode.
    """
    set_seed(0)

    out_folder += '/'  # !NOTE there has to be a / in the end

    cfg_bundletrack = yaml.load(open(f"{out_folder}/config_bundletrack.yml", 'r'))
    cfg_bundletrack['debug_dir'] = out_folder
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    yaml.dump(cfg_bundletrack, open(cfg_track_dir, 'w'))

    cfg_nerf = yaml.load(open(f"{out_folder}/config_nerf.yml", 'r'))
    cfg_nerf['n_step'] = 2000
    cfg_nerf['N_samples'] = 64
    cfg_nerf['N_samples_around_depth'] = 256
    cfg_nerf['first_frame_weight'] = 1
    cfg_nerf['down_scale_ratio'] = 1
    cfg_nerf['finest_res'] = 256
    cfg_nerf['num_levels'] = 16
    cfg_nerf['mesh_resolution'] = 0.002
    cfg_nerf['n_train_image'] = 500
    cfg_nerf['fs_sdf'] = 0.1
    cfg_nerf['frame_features'] = 2
    cfg_nerf['rgb_weight'] = 100

    cfg_nerf['i_img'] = np.inf
    cfg_nerf['i_mesh'] = cfg_nerf['i_img']
    cfg_nerf['i_nerf_normals'] = cfg_nerf['i_img']
    cfg_nerf['i_save_ray'] = cfg_nerf['i_img']

    cfg_nerf['datadir'] = f"{out_folder}/nerf_with_bundletrack_online"
    cfg_nerf['save_dir'] = copy.deepcopy(cfg_nerf['datadir'])

    os.makedirs(cfg_nerf['datadir'], exist_ok=True)

    cfg_nerf_dir = f"{cfg_nerf['datadir']}/config.yml"
    yaml.dump(cfg_nerf, open(cfg_nerf_dir, 'w'))

    # Use the passed video_dir instead of args.video_dir (which is now the root)
    reader = YcbineoatReader(video_dir=video_dir, downscale=1)

    tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5)
    tracker.cfg_nerf = cfg_nerf
    tracker.run_global_nerf(reader=reader, get_texture=True, tex_res=512)
    tracker.on_finish()

    print(f"Done with {out_folder}")


def postprocess_mesh(out_folder):
    mesh_files = sorted(glob.glob(f'{out_folder}/**/nerf/*normalized_space.obj', recursive=True))
    if not mesh_files:
        print(f"No normalized_space.obj found in {out_folder}")
        return

    print(f"Using {mesh_files[-1]}")
    os.makedirs(f"{out_folder}/mesh/", exist_ok=True)

    print(f"\nSaving meshes to {out_folder}/mesh/\n")

    mesh = trimesh.load(mesh_files[-1])
    with open(f'{os.path.dirname(mesh_files[-1])}/config.yml', 'r') as ff:
        cfg = yaml.load(ff)
    tf = np.eye(4)
    tf[:3, 3] = cfg['translation']
    tf1 = np.eye(4)
    tf1[:3, :3] *= cfg['sc_factor']
    tf = tf1 @ tf
    mesh.apply_transform(np.linalg.inv(tf))
    mesh.export(f"{out_folder}/mesh/mesh_real_scale.obj")

    components = trimesh_split(mesh, min_edge=1000)
    best_component = None
    best_size = 0
    for component in components:
        dists = np.linalg.norm(component.vertices, axis=-1)
        if len(component.vertices) > best_size:
            best_size = len(component.vertices)
            best_component = component
    
    if best_component is not None:
        mesh = trimesh_clean(best_component)
        mesh.export(f"{out_folder}/mesh/mesh_biggest_component.obj")
        mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=3, implicit_time_integration=False,
                                                  volume_constraint=True, laplacian_operator=None)
        mesh.export(f'{out_folder}/mesh/mesh_biggest_component_smoothed.obj')


def draw_pose(out_folder):
    K_path = f'{out_folder}/cam_K.txt'
    if not os.path.exists(K_path):
        print(f"Skipping draw_pose: {K_path} not found.")
        return

    K = np.loadtxt(K_path).reshape(3, 3)
    color_files = sorted(glob.glob(f'{out_folder}/color/*'))
    mesh_path = f'{out_folder}/textured_mesh.obj'
    
    if not os.path.exists(mesh_path):
        print(f"Skipping draw_pose: {mesh_path} not found.")
        return

    mesh = trimesh.load(mesh_path)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    out_dir = f'{out_folder}/pose_vis'
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Saving to {out_dir}")
    for color_file in color_files:
        color = imageio.imread(color_file)
        pose_path = color_file.replace('.png', '.txt').replace('color', 'ob_in_cam')
        if not os.path.exists(pose_path): continue
        
        pose = np.loadtxt(pose_path)
        pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, color, ob_in_cam=pose, bbox=bbox, line_color=(255, 255, 0))
        id_str = os.path.basename(color_file).replace('.png', '')
        imageio.imwrite(f'{out_dir}/{id_str}.png', vis)


def extract_pose_diff(out_folder):
    """
    Reads absolute pose data, calculates relative pose differences between consecutive frames.
    """
    # defines path
    pose_dir = f'{out_folder}/ob_in_cam'
    diff_out_dir = f'{out_folder}/pose_diff'
    diff_6d_out_dir = f'{out_folder}/pose_diff_6d'

    if not os.path.exists(pose_dir):
        print(f"Skipping pose diff: {pose_dir} not found.")
        return

    # create output directories
    os.makedirs(diff_out_dir, exist_ok=True)
    os.makedirs(diff_6d_out_dir, exist_ok=True)

    # get sorted list of pose files
    pose_files = sorted(glob.glob(f'{pose_dir}/*.txt'))

    print(f"Calculating pose differences and saving to {diff_out_dir} and {diff_6d_out_dir}")

    prev_pose = None
    for pose_file in pose_files:
        # load current absolute pose (4x4 matrix)
        curr_pose = np.loadtxt(pose_file)

        # calculate relative pose difference if previous pose exists
        if prev_pose is None:
            # first frame, no previous pose -> origin (identity matrix)
            pose_diff = np.eye(4)
        else:
            # T_diff = inv(T_prev) * T_curr
            pose_diff = np.linalg.inv(prev_pose) @ curr_pose

        # 1. save as 4x4 matrix
        filename = os.path.basename(pose_file)
        np.savetxt(f'{diff_out_dir}/{filename}', pose_diff, fmt='%.6f')

        # 2. convert to 6D vector (x, y, z, roll, pitch, yaw)
        translation = pose_diff[:3, 3]
        rotation_matrix = pose_diff[:3, :3]
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler('xyz', degrees=False)  # roll, pitch, yaw
        pose_6d = np.concatenate([translation, euler_angles])
        np.savetxt(f'{diff_6d_out_dir}/{filename}', pose_6d, fmt='%.6f')

        # update previous pose
        prev_pose = curr_pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="run_video", help="run_video/global_refine/draw_pose")
    parser.add_argument('--video_dir', type=str, default="/home/gavin/Documents/BundleSDF/demo_data/milk", 
                        help="Root directory containing subfolders of videos")
    parser.add_argument('--out_folder', type=str, default="/home/gavin/Documents/BundleSDF/demo_data/results_milk",
                        help="Root directory where output subfolders will be created")
    parser.add_argument('--use_segmenter', type=int, default=0)
    parser.add_argument('--use_gui', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1, help='interval of frames to run; 1 means using every frame')
    parser.add_argument('--debug_level', type=int, default=2, help='higher means more logging')
    args = parser.parse_args()

    # Get list of potential video directories
    if not os.path.exists(args.video_dir):
        print(f"Error: Input directory {args.video_dir} does not exist.")
        sys.exit(1)

    # Detect if args.video_dir is a single video or a root folder
    # We check if it contains a 'rgb' or 'color' folder directly
    is_single_video = os.path.exists(os.path.join(args.video_dir, 'rgb')) or \
                      os.path.exists(os.path.join(args.video_dir, 'color'))

    if is_single_video:
        video_dirs = [args.video_dir]
        print(f"Detected single video input.")
    else:
        # Assume it is a root directory containing multiple video subdirectories
        subdirs = sorted(os.listdir(args.video_dir))
        video_dirs = [os.path.join(args.video_dir, d) for d in subdirs if os.path.isdir(os.path.join(args.video_dir, d))]
        print(f"Detected batch input. Found {len(video_dirs)} directories.")

    if args.mode == 'run_video':
        for vid_path in video_dirs:
            # Determine specific output folder name
            folder_name = os.path.basename(os.path.normpath(vid_path))
            
            # If running in batch mode, append folder name to out_root.
            # If running single mode, user might have provided exact output path, but for consistency 
            # with the request, let's treat out_folder as root if we are processing a list.
            if is_single_video:
                current_out = args.out_folder
            else:
                current_out = os.path.join(args.out_folder, folder_name)

            try:
                run_one_video(
                    video_dir=vid_path, 
                    out_folder=current_out, 
                    use_segmenter=args.use_segmenter, 
                    use_gui=args.use_gui,
                    stride=args.stride,
                    debug_level=args.debug_level
                )
                
                # Optional: Run post-processing utilities automatically after success
                extract_pose_diff(current_out)
                
            except Exception as e:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"CRITICAL ERROR processing {vid_path}")
                print(f"Skipping this video and moving to the next.")
                print(f"Error details:")
                traceback.print_exc()
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue

    elif args.mode == 'global_refine':
        # Apply similar logic for global refine if needed in batch
        for vid_path in video_dirs:
            folder_name = os.path.basename(os.path.normpath(vid_path))
            if is_single_video:
                current_out = args.out_folder
            else:
                current_out = os.path.join(args.out_folder, folder_name)

            # In global_refine mode, out_folder must already exist
            if not os.path.exists(current_out):
                print(f"Skipping {current_out}, directory does not exist.")
                continue

            try:
                # Note: run_one_video_global_refine needs the source video dir and result dir
                run_one_video_global_nerf(video_dir=vid_path, out_folder=current_out)
            except Exception as e:
                print(f"Error refining {folder_name}: {e}")
                traceback.print_exc()

    elif args.mode == 'draw_pose':
        for vid_path in video_dirs:
            folder_name = os.path.basename(os.path.normpath(vid_path))
            if is_single_video:
                current_out = args.out_folder
            else:
                current_out = os.path.join(args.out_folder, folder_name)
            
            draw_pose(current_out)
    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")