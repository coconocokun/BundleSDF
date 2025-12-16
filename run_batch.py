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
import traceback
import copy
import multiprocessing  # Added for timeout handling
import time

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from segmentation_utils import Segmenter
from scipy.spatial.transform import Rotation as R


def run_one_video(video_dir, out_folder, use_segmenter=False, use_gui=False, stride=1, debug_level=2):
    """
    Runs BundleSDF tracking on a single video sequence.
    """
    # Re-seed inside the process to ensure consistency
    set_seed(0)
    
    print(f"--- Processing: {video_dir} ---")
    print(f"--- Outputting to: {out_folder} ---")

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

    # Pass specific video_dir to global nerf
    run_one_video_global_nerf(video_dir=video_dir, out_folder=out_folder)


def run_one_video_global_nerf(video_dir, out_folder):
    """
    Runs the global NeRF refinement step.
    """
    set_seed(0)

    out_folder += '/' 

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

    reader = YcbineoatReader(video_dir=video_dir, downscale=1)

    tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5)
    tracker.cfg_nerf = cfg_nerf
    tracker.run_global_nerf(reader=reader, get_texture=True, tex_res=512)
    tracker.on_finish()

    print(f"Done with {out_folder}")


def postprocess_mesh(out_folder):
    mesh_files = sorted(glob.glob(f'{out_folder}/**/nerf/*normalized_space.obj', recursive=True))
    if not mesh_files:
        return

    print(f"Using {mesh_files[-1]}")
    os.makedirs(f"{out_folder}/mesh/", exist_ok=True)

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
    if not os.path.exists(K_path): return
    K = np.loadtxt(K_path).reshape(3, 3)
    
    color_files = sorted(glob.glob(f'{out_folder}/color/*'))
    mesh_path = f'{out_folder}/textured_mesh.obj'
    if not os.path.exists(mesh_path): return

    mesh = trimesh.load(mesh_path)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    out_dir = f'{out_folder}/pose_vis'
    os.makedirs(out_dir, exist_ok=True)
    
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
    pose_dir = f'{out_folder}/ob_in_cam'
    diff_out_dir = f'{out_folder}/pose_diff'
    diff_6d_out_dir = f'{out_folder}/pose_diff_6d'

    if not os.path.exists(pose_dir): return

    os.makedirs(diff_out_dir, exist_ok=True)
    os.makedirs(diff_6d_out_dir, exist_ok=True)
    pose_files = sorted(glob.glob(f'{pose_dir}/*.txt'))

    prev_pose = None
    for pose_file in pose_files:
        curr_pose = np.loadtxt(pose_file)
        if prev_pose is None:
            pose_diff = np.eye(4)
        else:
            pose_diff = np.linalg.inv(prev_pose) @ curr_pose
        
        filename = os.path.basename(pose_file)
        np.savetxt(f'{diff_out_dir}/{filename}', pose_diff, fmt='%.6f')
        
        translation = pose_diff[:3, 3]
        rotation_matrix = pose_diff[:3, :3]
        r = R.from_matrix(rotation_matrix)
        euler_angles = r.as_euler('xyz', degrees=False)
        pose_6d = np.concatenate([translation, euler_angles])
        np.savetxt(f'{diff_6d_out_dir}/{filename}', pose_6d, fmt='%.6f')
        prev_pose = curr_pose


def wrapper_run_one_video(video_dir, out_folder, use_segmenter, use_gui, stride, debug_level):
    """
    Wrapper to handle exceptions inside the process so they are reported before the process dies.
    """
    try:
        run_one_video(video_dir, out_folder, use_segmenter, use_gui, stride, debug_level)
    except Exception as e:
        print(f"Error in process for {video_dir}:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Necessary for multiprocessing to work on some platforms
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="run_video", help="run_video/global_refine/draw_pose")
    parser.add_argument('--video_dir', type=str, default="/home/gavin/Documents/BundleSDF/demo_data/milk")
    parser.add_argument('--out_folder', type=str, default="/home/gavin/Documents/BundleSDF/demo_data/results_milk")
    parser.add_argument('--use_segmenter', type=int, default=0)
    parser.add_argument('--use_gui', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1, help='interval of frames to run')
    parser.add_argument('--debug_level', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=3600, help='Max seconds per video before skipping')
    args = parser.parse_args()

    if not os.path.exists(args.video_dir):
        print(f"Error: Input directory {args.video_dir} does not exist.")
        sys.exit(1)

    # Detect single video or batch directory
    is_single_video = os.path.exists(os.path.join(args.video_dir, 'rgb')) or \
                      os.path.exists(os.path.join(args.video_dir, 'color'))

    if is_single_video:
        video_dirs = [args.video_dir]
    else:
        subdirs = sorted(os.listdir(args.video_dir))
        video_dirs = [os.path.join(args.video_dir, d) for d in subdirs if os.path.isdir(os.path.join(args.video_dir, d))]
        print(f"Detected batch input. Found {len(video_dirs)} directories.")

    if args.mode == 'run_video':
        for vid_path in video_dirs:
            folder_name = os.path.basename(os.path.normpath(vid_path))
            current_out = args.out_folder if is_single_video else os.path.join(args.out_folder, folder_name)

            print(f"\n=======================================================")
            print(f"Starting process for: {folder_name}")
            print(f"Timeout set to: {args.timeout} seconds")
            print(f"=======================================================\n")

            # Create a separate process for the heavy lifting
            p = multiprocessing.Process(
                target=wrapper_run_one_video,
                args=(vid_path, current_out, args.use_segmenter, args.use_gui, args.stride, args.debug_level)
            )
            
            p.start()
            
            # Wait for the process to finish or timeout
            p.join(timeout=args.timeout)

            if p.is_alive():
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"TIMEOUT REACHED ({args.timeout}s) for {folder_name}")
                print(f"Killing process and skipping to next video...")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                p.terminate()
                # Give it a second to die gracefully, otherwise force kill
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
                
                # Optional: Cleanup the failed output folder to save space/confusion
                # os.system(f'rm -rf {current_out}') 
            elif p.exitcode != 0:
                 print(f"Process for {folder_name} failed with exit code {p.exitcode}. check logs above.")
            else:
                print(f"Process for {folder_name} finished successfully.")
                # Run lightweight post-processing in the main process
                extract_pose_diff(current_out)

    elif args.mode == 'global_refine':
        # Similar logic for refine if needed, or keep simple if refine is fast
        for vid_path in video_dirs:
            folder_name = os.path.basename(os.path.normpath(vid_path))
            current_out = args.out_folder if is_single_video else os.path.join(args.out_folder, folder_name)
            
            if not os.path.exists(current_out): continue
            try:
                run_one_video_global_nerf(video_dir=vid_path, out_folder=current_out)
            except Exception:
                traceback.print_exc()

    elif args.mode == 'draw_pose':
        for vid_path in video_dirs:
            folder_name = os.path.basename(os.path.normpath(vid_path))
            current_out = args.out_folder if is_single_video else os.path.join(args.out_folder, folder_name)
            draw_pose(current_out)