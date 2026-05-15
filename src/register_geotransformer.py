"""
Copyright (c) 2026 Shibin Tang, Dalian University of Technology.
Contact: tang_Shibin@dlut.edu.cn

Licensed under the MIT License. See the LICENSE file in the project root
for full license information.

--------------------------------------------------------------------------------
Batch evaluation script for GeoTransformer-based point cloud registration.

For each pair listed in the configuration file, this script:
  1. Loads the source / target clouds and the ground-truth transform.
  2. Voxel-downsamples both clouds.
  3. Runs GeoTransformer inference with a pre-trained checkpoint
     (3DMatch or KITTI) to predict dense correspondences and an estimated
     rigid transform.
  4. Computes the standard registration metrics.

Evaluation metrics:
  - RRE  : Relative Rotation Error    (deg)
  - RTE  : Relative Translation Error (m)
  - FMR  : Feature Matching Recall  (FM_IR > threshold is a correct match)
  - RR   : Registration Recall      (RTE < threshold AND RRE < threshold)
  - Time : Seconds per 1000 keypoints (s/KPts)

The GeoTransformer source code and pre-trained weights are expected
under `third_party/GeoTransformer/` (provided as a git submodule). You
can override the location with --geotransformer_root.
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import time
import numpy as np
import torch
import open3d as o3d


def make_open3d_point_cloud(xyz, color=None):
    """Create an Open3D point cloud from an Nx3 numpy array."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def load_point_cloud(file_path):
    """
    Load a point cloud from disk.

    Supported formats: .ply, .pcd, .xyz, .pts, .txt, .npy, .bin.
    Only XYZ coordinates are returned (any extra channels are dropped).
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.ply', '.pcd']:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    elif ext == '.npy':
        points = np.load(file_path)
        if points.shape[1] > 3:
            points = points[:, :3]
    elif ext == '.bin':
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    elif ext in ['.xyz', '.pts', '.txt']:
        points = np.loadtxt(file_path)
        if points.shape[1] > 3:
            points = points[:, :3]
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return points.astype(np.float32)


def voxel_downsample(points, voxel_size=0.3):
    """Uniform voxel downsampling via Open3D."""
    pcd = make_open3d_point_cloud(points)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_down.points).astype(np.float32)


def compute_mean_std(data):
    """Return (mean, unbiased_std). Returns (0, 0) for empty input."""
    if len(data) == 0:
        return 0.0, 0.0

    mean = np.mean(data)
    if len(data) < 2:
        return mean, 0.0

    std = np.std(data, ddof=1)
    return mean, std


def compute_registration_rmse(source_points, gt_transform, est_transform):
    """
    Transform-difference RMSE:
        RMSE = sqrt( mean( || T_gt * p_i - T_est * p_i ||^2 ) ).
    """
    gt_transformed = (gt_transform[:3, :3] @ source_points.T).T + gt_transform[:3, 3]
    est_transformed = (est_transform[:3, :3] @ source_points.T).T + est_transform[:3, 3]

    errors = np.linalg.norm(gt_transformed - est_transformed, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))

    return rmse


def compute_all_correspondence_metrics(ref_corr_points, src_corr_points,
                                       gt_transform, est_transform,
                                       default_thresh, distance_thresholds):
    """
    Jointly compute correspondence-based metrics in a single pass:
      - fm_ir            : feature-matching inlier ratio (validated by T_gt)
      - reg_ir           : post-registration inlier ratio (validated by T_est)
      - fm_ir_by_distance: fm_ir for several distance thresholds
    """
    if len(ref_corr_points) == 0:
        return 0.0, 0.0, {thresh: 0.0 for thresh in distance_thresholds}

    num_corr = len(ref_corr_points)

    src_gt_transformed = (gt_transform[:3, :3] @ src_corr_points.T).T + gt_transform[:3, 3]
    gt_distances = np.linalg.norm(ref_corr_points - src_gt_transformed, axis=1)

    src_est_transformed = (est_transform[:3, :3] @ src_corr_points.T).T + est_transform[:3, 3]
    est_distances = np.linalg.norm(ref_corr_points - src_est_transformed, axis=1)

    fm_ir = np.sum(gt_distances < default_thresh) / num_corr
    reg_ir = np.sum(est_distances < default_thresh) / num_corr

    fm_ir_by_distance = {}
    for thresh in distance_thresholds:
        fm_ir_by_distance[thresh] = np.sum(gt_distances < thresh) / num_corr

    return fm_ir, reg_ir, fm_ir_by_distance


def parse_config_file(config_path, data_dir):
    """
    Parse the pair configuration file.

    Expected per-block format:
        <source_id>-<target_id>
        r00 r01 r02 tx
        r10 r11 r12 ty
        r20 r21 r22 tz
        0   0   0   1
        <blank line>

    Numeric IDs are zero-padded to 3 digits to match the
    `Scan_<id>.pcd` file naming convention.
    """
    pairs = []

    with open(config_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith('#') or line.startswith(';'):
            i += 1
            continue

        if '-' in line and line[0].isdigit():
            parts = line.split('-')
            if len(parts) == 2:
                try:
                    source_id = parts[0].strip()
                    target_id = parts[1].strip()

                    source_id_padded = source_id.zfill(3)
                    target_id_padded = target_id.zfill(3)

                    source_path = os.path.join(data_dir, f"Scan_{source_id_padded}.pcd")
                    target_path = os.path.join(data_dir, f"Scan_{target_id_padded}.pcd")

                    gt_transform = np.zeros((4, 4))
                    for j in range(4):
                        i += 1
                        if i < len(lines):
                            row_values = [float(x) for x in lines[i].strip().split()]
                            gt_transform[j, :len(row_values)] = row_values

                    if os.path.exists(source_path) and os.path.exists(target_path):
                        pairs.append({
                            'source_id': source_id,
                            'target_id': target_id,
                            'source_path': source_path,
                            'target_path': target_path,
                            'gt_transform': gt_transform,
                        })
                    else:
                        print(f"  Warning: Files not found for pair {source_id}-{target_id}")
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Failed to parse line: {line}, error: {e}")

        i += 1

    return pairs


def save_results_to_config(results, output_path):
    """Write estimated transforms in the same format as the input config file."""
    with open(output_path, 'w') as f:
        for result in results:
            f.write(f"{result['source_id']}-{result['target_id']}\n")
            T = result['est_transform']
            for i in range(4):
                row = ' '.join([f"{T[i, j]:.6f}" for j in range(4)])
                f.write(f"{row}\n")
            f.write("\n")


def print_results_table(metrics):
    """Pretty-print the aggregate metrics."""
    print("\n" + "=" * 80)
    print("GeoTransformer Registration Evaluation Results")
    print("=" * 80)

    print(f"{'Metric':<20} {'Value':<25} {'Std':<20}")
    print("-" * 65)

    rre_mean, rre_std = metrics['rre']
    print(f"{'RRE (deg)':<20} {rre_mean:<25.4f} +/- {rre_std:.4f}")

    rte_mean, rte_std = metrics['rte']
    print(f"{'RTE (m)':<20} {rte_mean:<25.4f} +/- {rte_std:.4f}")

    rmse_mean, rmse_std = metrics['rmse']
    print(f"{'RMSE (m)':<20} {rmse_mean:<25.4f} +/- {rmse_std:.4f}")

    reg_ir_mean, reg_ir_std = metrics['reg_ir']
    print(f"{'Reg_IR (%)':<20} {reg_ir_mean * 100:<25.2f} +/- {reg_ir_std * 100:.2f}")

    fmr_rate = metrics['fmr']
    print(f"{'FMR (%)':<20} {fmr_rate * 100:<25.2f} ({metrics['feature_matches']}/{metrics['total_pairs']})")

    rr_rate = metrics['rr']
    print(f"{'RR (%)':<20} {rr_rate * 100:<25.2f} ({metrics['successful_registrations']}/{metrics['total_pairs']})")

    time_per_kpts = metrics['time_per_kpts']
    print(f"{'Time (s/KPts)':<20} {time_per_kpts:<25.4f}")

    print("=" * 80)

    print(f"\nTotal pairs evaluated: {metrics['total_pairs']}")
    print(f"Successful registrations (RR): {metrics['successful_registrations']}/{metrics['total_pairs']}")
    print(f"Feature matches (FMR): {metrics['feature_matches']}/{metrics['total_pairs']}")
    print(f"Total keypoints processed: {metrics['total_keypoints']}")
    print(f"Total time: {metrics['total_time']:.2f}s")


def print_fmr_tables(fm_ir_list, fm_ir_by_distance, fm_ir_thresholds,
                     distance_thresholds, default_ir_thresh):
    """Print FMR sweeps over the IR threshold and the distance threshold."""
    # FMR vs FM_IR threshold (fixed distance threshold)
    print("\n" + "=" * 80)
    print("FMR vs FM_IR Threshold (Fixed Distance Threshold)")
    print("=" * 80)
    print(f"{'FM_IR Thresh':<15} {'FMR (%)':<20} {'Matches':<15}")
    print("-" * 50)

    fm_ir_array = np.array(fm_ir_list)
    total_pairs = len(fm_ir_list)

    for thresh in fm_ir_thresholds:
        fmr_success = (fm_ir_array > thresh).astype(int)
        matches = np.sum(fmr_success)
        fmr_rate = matches / total_pairs if total_pairs > 0 else 0
        print(f"{thresh * 100:>10.2f}%    {fmr_rate * 100:<20.2f} {matches}/{total_pairs}")

    print("=" * 80)

    # FMR vs distance threshold (fixed FM_IR threshold)
    print("\n" + "=" * 80)
    print(f"FMR vs Distance Threshold (Fixed FM_IR Threshold = {default_ir_thresh * 100:.1f}%)")
    print("=" * 80)
    print(f"{'Dist Thresh (m)':<18} {'FMR (%)':<20} {'Matches':<15}")
    print("-" * 53)

    for dist_thresh in distance_thresholds:
        if dist_thresh in fm_ir_by_distance:
            fm_ir_list_d = fm_ir_by_distance[dist_thresh]
            fm_ir_array_d = np.array(fm_ir_list_d)
            fmr_success = (fm_ir_array_d > default_ir_thresh).astype(int)
            matches = np.sum(fmr_success)
            fmr_rate = matches / len(fm_ir_list_d) if len(fm_ir_list_d) > 0 else 0
            print(f"{dist_thresh:>12.2f}      {fmr_rate * 100:<20.2f} {matches}/{len(fm_ir_list_d)}")

    print("=" * 80)


def load_geotransformer_model(geotransformer_root, model_name):
    """
    Resolve the GeoTransformer config / weights for the requested checkpoint
    (3DMatch or KITTI), import the corresponding `make_cfg` and `create_model`,
    then load the pre-trained weights onto CUDA.

    Returns: (model, cfg, neighbor_limits, weights_path).
    """
    sys.path.insert(0, geotransformer_root)

    if model_name == 'KITTI':
        config_subpath = os.path.join(
            geotransformer_root,
            'experiments',
            'geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn',
        )
        weights_path = os.path.join(geotransformer_root, 'weights', 'geotransformer-kitti.pth.tar')
        neighbor_limits = [39, 35, 35, 38, 38]  # KITTI defaults
    else:
        config_subpath = os.path.join(
            geotransformer_root,
            'experiments',
            'geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn',
        )
        weights_path = os.path.join(geotransformer_root, 'weights', 'geotransformer-3dmatch.pth.tar')
        neighbor_limits = [38, 36, 36, 38]      # 3DMatch defaults

    if not os.path.isdir(config_subpath):
        raise FileNotFoundError(
            f"Could not find GeoTransformer experiment at: {config_subpath}\n"
            f"Did you initialise the GeoTransformer submodule? See README.md."
        )
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Could not find GeoTransformer weights at: {weights_path}\n"
            f"Please download them following the GeoTransformer repository instructions."
        )

    sys.path.insert(0, config_subpath)
    from config import make_cfg
    from model import create_model

    cfg = make_cfg()
    model = create_model(cfg).cuda()
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict["model"])
    model.eval()

    return model, cfg, neighbor_limits, weights_path


def main():
    parser = argparse.ArgumentParser(description='GeoTransformer Point Cloud Registration Batch Evaluation')
    parser.add_argument('--data_dir', type=str, default='./EvalData',
                        help='Directory containing point cloud files')
    parser.add_argument('--config', type=str, default='./EvalData/config.txt',
                        help='Path to pairs configuration file')
    parser.add_argument('--output', type=str, default='./EvalData/geotransformer_results.txt',
                        help='Path to save estimated transforms')
    parser.add_argument('--model', type=str, choices=['3DMatch', 'KITTI'], default='KITTI',
                        help='Pre-trained model to use (3DMatch for indoor, KITTI for outdoor)')
    parser.add_argument('--geotransformer_root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', 'third_party', 'GeoTransformer'),
                        help='Path to the GeoTransformer source tree (experiments/ and weights/)')
    parser.add_argument('--voxel_size', type=float, default=0.2,
                        help='Voxel size for downsampling (meters)')
    parser.add_argument('--fmr_threshold', type=float, default=0.005,
                        help='Inlier ratio threshold for FMR')
    parser.add_argument('--rr_rte_threshold', type=float, default=0.1,
                        help='RTE threshold for RR (meters)')
    parser.add_argument('--rr_rre_threshold', type=float, default=5.0,
                        help='RRE threshold for RR (degrees)')
    parser.add_argument('--inlier_distance', type=float, default=0.15,
                        help='Distance threshold for inlier computation (meters)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--fm_ir_thresholds', type=str,
                        default='0.0005,0.001,0.002,0.003,0.005,0.01,0.02,0.03,0.05',
                        help='Comma-separated FM_IR thresholds for FMR analysis')
    parser.add_argument('--distance_thresholds', type=str,
                        default='0.01,0.02,0.03,0.05,0.1,0.15,0.2,0.3,0.5',
                        help='Comma-separated distance thresholds for FM_IR analysis (meters)')

    args = parser.parse_args()

    fm_ir_thresholds = [float(x) for x in args.fm_ir_thresholds.split(',')]
    distance_thresholds = [float(x) for x in args.distance_thresholds.split(',')]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    geotr_root = os.path.abspath(args.geotransformer_root)

    print("=" * 80)
    print("GeoTransformer Point Cloud Registration Batch Evaluation")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Config file: {args.config}")
    print(f"  Model: {args.model}")
    print(f"  GeoTransformer root: {geotr_root}")
    print(f"  Voxel size: {args.voxel_size}m")

    # Import GeoTransformer utilities. These rely on the submodule being
    # initialised AND installed (pip install -e third_party/GeoTransformer).
    if geotr_root not in sys.path:
        sys.path.insert(0, geotr_root)
    from geotransformer.utils.data import registration_collate_fn_stack_mode
    from geotransformer.utils.torch import to_cuda, release_cuda
    from geotransformer.utils.registration import compute_registration_error

    print(f"\n[1/4] Parsing configuration file...")
    pairs = parse_config_file(args.config, args.data_dir)
    print(f"  Found {len(pairs)} valid point cloud pairs")

    if len(pairs) == 0:
        print("Error: No valid pairs found!")
        return

    print(f"\n[2/4] Loading {args.model} pre-trained model...")
    model, cfg, neighbor_limits, weights_path = load_geotransformer_model(geotr_root, args.model)
    print(f"  Model loaded from: {weights_path}")

    print(f"\n[3/4] Evaluating {len(pairs)} pairs...")

    results = []
    rre_list = []
    rte_list = []
    fm_ir_list = []
    reg_ir_list = []
    reg_rmse_list = []
    fm_ir_by_distance = {d: [] for d in distance_thresholds}
    fmr_success = []
    rr_success = []
    total_time = 0.0
    total_keypoints = 0

    for idx, pair in enumerate(pairs):
        print(f"\n  [{idx + 1}/{len(pairs)}] Processing {pair['source_id']}-{pair['target_id']}...")

        try:
            src_points = load_point_cloud(pair['source_path'])
            ref_points = load_point_cloud(pair['target_path'])
            print(f"    Loaded: source={src_points.shape[0]} pts, target={ref_points.shape[0]} pts")

            src_points_down = voxel_downsample(src_points, args.voxel_size)
            ref_points_down = voxel_downsample(ref_points, args.voxel_size)
            print(f"    After voxel downsampling: source={src_points_down.shape[0]} pts, "
                  f"target={ref_points_down.shape[0]} pts")

            num_keypts = src_points_down.shape[0] + ref_points_down.shape[0]
            total_keypoints += num_keypts

            src_feats = np.ones_like(src_points_down[:, :1])
            ref_feats = np.ones_like(ref_points_down[:, :1])

            data_dict = {
                "ref_points": ref_points_down.astype(np.float32),
                "src_points": src_points_down.astype(np.float32),
                "ref_feats": ref_feats.astype(np.float32),
                "src_feats": src_feats.astype(np.float32),
                "transform": pair['gt_transform'].astype(np.float32),
            }

            start_time = time.time()

            data_dict = registration_collate_fn_stack_mode(
                [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
                cfg.backbone.init_radius, neighbor_limits,
            )

            data_dict = to_cuda(data_dict)
            with torch.no_grad():
                output_dict = model(data_dict)
            data_dict = release_cuda(data_dict)
            output_dict = release_cuda(output_dict)

            pair_time = time.time() - start_time
            total_time += pair_time

            est_transform = output_dict["estimated_transform"]
            gt_transform = pair['gt_transform']

            rre, rte = compute_registration_error(gt_transform, est_transform)
            rre_list.append(rre)
            rte_list.append(rte)

            ref_corr_points = output_dict.get("ref_corr_points", np.array([]))
            src_corr_points = output_dict.get("src_corr_points", np.array([]))
            num_correspondences = len(ref_corr_points)

            fm_ir, reg_ir, fm_ir_multi = compute_all_correspondence_metrics(
                ref_corr_points, src_corr_points,
                gt_transform, est_transform,
                args.inlier_distance, distance_thresholds,
            )
            fm_ir_list.append(fm_ir)
            reg_ir_list.append(reg_ir)
            for d in distance_thresholds:
                fm_ir_by_distance[d].append(fm_ir_multi[d])

            reg_rmse = compute_registration_rmse(src_points_down, gt_transform, est_transform)
            reg_rmse_list.append(reg_rmse)

            fmr_success.append(1 if fm_ir > args.fmr_threshold else 0)
            is_success = (rte < args.rr_rte_threshold) and (rre < args.rr_rre_threshold)
            rr_success.append(1 if is_success else 0)

            print(f"    Correspondences: {num_correspondences}")
            print(f"    FM_IR: {fm_ir * 100:.2f}% (GT), Reg_IR: {reg_ir * 100:.2f}% (Est), "
                  f"RMSE: {reg_rmse:.4f} m")
            print(f"    RRE: {rre:.4f} deg, RTE: {rte:.4f} m, Time: {pair_time:.2f}s")
            print(f"    -> FM_IR>{args.fmr_threshold * 100:.1f}%: "
                  f"{'Yes' if fmr_success[-1] else 'No'}, "
                  f"RTE<{args.rr_rte_threshold}m & RRE<{args.rr_rre_threshold}deg: "
                  f"{'Yes' if rr_success[-1] else 'No'}")

            results.append({
                'source_id': pair['source_id'],
                'target_id': pair['target_id'],
                'est_transform': est_transform,
                'gt_transform': gt_transform,
                'rre': rre,
                'rte': rte,
                'rmse': reg_rmse,
                'fm_ir': fm_ir,
                'reg_ir': reg_ir,
                'num_correspondences': num_correspondences,
                'time': pair_time,
            })

        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n[4/4] Computing statistics...")

    rre_mean, rre_std = compute_mean_std(rre_list)
    rte_mean, rte_std = compute_mean_std(rte_list)
    rmse_mean, rmse_std = compute_mean_std(reg_rmse_list)
    reg_ir_mean, reg_ir_std = compute_mean_std(reg_ir_list)
    fmr_rate = sum(fmr_success) / len(fmr_success) if len(fmr_success) > 0 else 0
    rr_rate = sum(rr_success) / len(rr_success) if len(rr_success) > 0 else 0

    time_per_kpts = (total_time / total_keypoints) * 1000 if total_keypoints > 0 else 0

    metrics = {
        'rre': (rre_mean, rre_std),
        'rte': (rte_mean, rte_std),
        'rmse': (rmse_mean, rmse_std),
        'reg_ir': (reg_ir_mean, reg_ir_std),
        'fmr': fmr_rate,
        'rr': rr_rate,
        'time_per_kpts': time_per_kpts,
        'total_pairs': len(pairs),
        'successful_registrations': sum(rr_success),
        'feature_matches': sum(fmr_success),
        'total_keypoints': total_keypoints,
        'total_time': total_time,
    }

    print_results_table(metrics)
    print_fmr_tables(fm_ir_list, fm_ir_by_distance, fm_ir_thresholds,
                     distance_thresholds, args.fmr_threshold)

    save_results_to_config(results, args.output)
    print(f"\nEstimated transforms saved to: {args.output}")

    csv_path = args.output.replace('.txt', '_detailed.csv')
    with open(csv_path, 'w') as f:
        f.write("source_id,target_id,RRE(deg),RTE(m),RMSE(m),FM_IR(%),Reg_IR(%),"
                "FMR,RR,correspondences,time(s)\n")
        for i, result in enumerate(results):
            f.write(f"{result['source_id']},{result['target_id']},"
                    f"{result['rre']:.6f},{result['rte']:.6f},")
            f.write(f"{result['rmse']:.6f},{result['fm_ir'] * 100:.4f},"
                    f"{result['reg_ir'] * 100:.4f},{fmr_success[i]},{rr_success[i]},")
            f.write(f"{result['num_correspondences']},{result['time']:.4f}\n")
    print(f"Detailed results saved to: {csv_path}")

    return metrics


if __name__ == '__main__':
    main()
