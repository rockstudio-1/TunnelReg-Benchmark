/**
 * Copyright (c) 2026 Shibin Tang, Dalian University of Technology.
 * Contact: tang_Shibin@dlut.edu.cn
 *
 * Licensed under the MIT License. See the LICENSE file in the project root
 * for full license information.
 *
 * ----------------------------------------------------------------------------
 * Batch evaluation program for classical point cloud registration methods.
 *
 * Methods evaluated:
 *   1. FPFH descriptor + RANSAC
 *   2. CSHOT (Color-SHOT) descriptor + RANSAC
 *   3. ICP
 *
 * Evaluation metrics:
 *   - RRE  (Relative Rotation Error,    degrees)
 *   - RTE  (Relative Translation Error, meters)
 *   - FMR  (Feature Matching Recall): a pair is regarded as successfully
 *          matched when its inlier ratio (IR) is above the threshold
 *          (default 5%).
 *   - RR   (Registration Recall): a pair is regarded as successfully
 *          registered when RTE < 0.1 m AND RRE < 5 deg.
 *   - Time: seconds per 1000 points (s/KPts), i.e. the average registration
 *          time normalised by point count.
 * ----------------------------------------------------------------------------
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <map>
#include <tuple>

#include <pcl/point_types.h>
#include <pcl/console/print.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/correspondence.h>

// ============================================================================
// Point cloud registration pair
// ============================================================================
struct PointCloudPair
{
    std::string source_id;              // Source point cloud ID (e.g. "182")
    std::string target_id;              // Target point cloud ID (e.g. "183")
    Eigen::Matrix4f gt_matrix;          // Ground-truth transform (source -> target)
};

// ============================================================================
// Configuration parameters
// ============================================================================
struct EvalConfig
{
    // Data
    std::string data_directory;             // Point cloud data directory
    std::string pairs_config_file;          // Pair configuration filename (e.g. "config.txt")

    // Evaluation
    float inlier_distance_threshold;        // Inlier distance threshold (m)
    float fmr_inlier_ratio_threshold;       // FMR judgement threshold (IR > value -> match correct)
    float rr_rte_threshold;                 // RR judgement: max RTE (m)
    float rr_rre_threshold;                 // RR judgement: max RRE (deg)

    // Preprocessing
    float voxel_grid_leaf_size;             // Voxel grid leaf size

    // Normal estimation
    int normal_k_search;

    // Feature computation
    float feature_radius;                   // Feature search radius (0 -> auto)
    float auto_radius_multiplier;           // Multiplier on median neighbour distance

    // RANSAC
    int ransac_max_iterations;
    int ransac_nr_samples;
    float ransac_min_sample_distance;
    float ransac_similarity_threshold;
    float ransac_max_correspondence_distance;
    float ransac_inlier_fraction;

    // ICP
    float icp_max_correspondence_distance;
    int icp_max_iterations;
    double icp_transformation_epsilon;
    double icp_euclidean_fitness_epsilon;

    // Output
    std::string output_directory;
    bool save_aligned_clouds;

    // Visualization
    bool enable_visualization;

    // Metric toggles
    bool compute_inlier_ratio;              // Whether to compute IR
    bool compute_metrics;                   // Whether to compute RRE/RTE/RMSE/Reg_IR

    EvalConfig()
        : data_directory(""),
          pairs_config_file("config.txt"),
          inlier_distance_threshold(0.05f),
          fmr_inlier_ratio_threshold(0.05f),   // 5%
          rr_rte_threshold(0.1f),              // 0.1 m
          rr_rre_threshold(5.0f),              // 5 deg
          voxel_grid_leaf_size(0.1f),
          normal_k_search(30),
          feature_radius(0.0f),                // 0 -> auto
          auto_radius_multiplier(7.5f),        // 7.5 x median neighbour distance
          ransac_max_iterations(5000),
          ransac_nr_samples(3),
          ransac_min_sample_distance(0.05f),
          ransac_similarity_threshold(0.9f),
          ransac_max_correspondence_distance(0.5f),
          ransac_inlier_fraction(0.25f),
          icp_max_correspondence_distance(0.1f),
          icp_max_iterations(50),
          icp_transformation_epsilon(1e-8),
          icp_euclidean_fitness_epsilon(1e-6),
          output_directory(""),
          save_aligned_clouds(false),
          enable_visualization(false),
          compute_inlier_ratio(true),
          compute_metrics(true)
    {
    }
};

// ============================================================================
// Single-pair evaluation result
// ============================================================================
struct EvalResult
{
    std::string method_name;
    std::string pair_id;                // e.g. "182-183"
    double rre;                         // Relative rotation error (deg)
    double rte;                         // Relative translation error (m)
    double inlier_ratio;                // Feature-matching inlier ratio (validated by GT)
    double registration_ir;             // Post-registration inlier ratio (overlap between
                                        // transformed source and target)
    double inlier_rmse;                 // Registration RMSE (m), based on T_est vs T_gt
    double time_ms;                     // Elapsed time (ms)
    int num_points;                     // Point count (for s/KPts)
    int num_correspondences;            // #correspondences
    int num_inliers;                    // #inliers
    Eigen::Matrix4f transformation;     // Estimated transformation
    bool success;                       // Registration converged
    bool fmr_success;                   // IR > threshold
    bool rr_success;                    // RTE & RRE both below thresholds

    // For multi-threshold analysis: per-correspondence GT distances.
    std::vector<float> correspondence_distances;

    EvalResult() : rre(0), rte(0), inlier_ratio(0), registration_ir(-1.0), inlier_rmse(-1.0), time_ms(0),
                   num_points(0), num_correspondences(0), num_inliers(0),
                   transformation(Eigen::Matrix4f::Identity()),
                   success(false), fmr_success(false), rr_success(false) {}
};

// ============================================================================
// Per-method aggregate statistics
// ============================================================================
struct MethodStats
{
    std::string method_name;
    double avg_rre;
    double avg_rte;
    double avg_inlier_rmse;
    double avg_registration_ir;
    double fmr;
    double rr;
    double time_per_kpts;               // s / KPts

    // Standard deviations (FMR and RR are rates; no std)
    double std_rre;
    double std_rte;
    double std_inlier_rmse;
    double std_registration_ir;

    int total_pairs;
    int fmr_count;
    int rr_count;
    double total_time_ms;
    int total_points;

    MethodStats() : avg_rre(0), avg_rte(0), avg_inlier_rmse(0), avg_registration_ir(0),
                    fmr(0), rr(0), time_per_kpts(0),
                    std_rre(0), std_rte(0), std_inlier_rmse(0), std_registration_ir(0),
                    total_pairs(0), fmr_count(0), rr_count(0),
                    total_time_ms(0), total_points(0) {}
};

// ============================================================================
// Configuration file loader (simple key=value, ignores '#' and ';' comments)
// ============================================================================
bool loadConfig(const std::string &config_file, EvalConfig &config)
{
    std::ifstream file(config_file);
    if (!file.is_open())
    {
        std::cerr << "Failed to open config file: " << config_file << std::endl;
        return false;
    }

    std::string line;

    while (std::getline(file, line))
    {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';')
            continue;

        // Skip matrix markers
        if (line[0] == '[')
            continue;

        // key=value
        size_t pos = line.find('=');
        if (pos == std::string::npos)
            continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));

        // Strip surrounding quotes if any
        if (!value.empty() && (value[0] == '"' || value[0] == '\''))
        {
            char quote = value[0];
            size_t end_quote = value.find(quote, 1);
            if (end_quote != std::string::npos)
                value = value.substr(1, end_quote - 1);
            else
                value = value.substr(1);
        }
        else
        {
            value.erase(value.find_last_not_of(" \t") + 1);
        }

        try
        {
            if (key == "data_directory") config.data_directory = value;
            else if (key == "pairs_config_file") config.pairs_config_file = value;
            else if (key == "inlier_distance_threshold") config.inlier_distance_threshold = std::stof(value);
            else if (key == "fmr_inlier_ratio_threshold") config.fmr_inlier_ratio_threshold = std::stof(value);
            else if (key == "rr_rte_threshold") config.rr_rte_threshold = std::stof(value);
            else if (key == "rr_rre_threshold") config.rr_rre_threshold = std::stof(value);
            else if (key == "voxel_grid_leaf_size") config.voxel_grid_leaf_size = std::stof(value);
            else if (key == "normal_k_search") config.normal_k_search = std::stoi(value);
            else if (key == "feature_radius") config.feature_radius = std::stof(value);
            else if (key == "auto_radius_multiplier") config.auto_radius_multiplier = std::stof(value);
            else if (key == "ransac_max_iterations") config.ransac_max_iterations = std::stoi(value);
            else if (key == "ransac_nr_samples") config.ransac_nr_samples = std::stoi(value);
            else if (key == "ransac_min_sample_distance") config.ransac_min_sample_distance = std::stof(value);
            else if (key == "ransac_similarity_threshold") config.ransac_similarity_threshold = std::stof(value);
            else if (key == "ransac_max_correspondence_distance") config.ransac_max_correspondence_distance = std::stof(value);
            else if (key == "ransac_inlier_fraction") config.ransac_inlier_fraction = std::stof(value);
            else if (key == "icp_max_correspondence_distance") config.icp_max_correspondence_distance = std::stof(value);
            else if (key == "icp_max_iterations") config.icp_max_iterations = std::stoi(value);
            else if (key == "icp_transformation_epsilon") config.icp_transformation_epsilon = std::stod(value);
            else if (key == "icp_euclidean_fitness_epsilon") config.icp_euclidean_fitness_epsilon = std::stod(value);
            else if (key == "output_directory") config.output_directory = value;
            else if (key == "save_aligned_clouds") config.save_aligned_clouds = (value == "1" || value == "true");
            else if (key == "enable_visualization") config.enable_visualization = (value == "1" || value == "true");
            else if (key == "compute_inlier_ratio") config.compute_inlier_ratio = (value == "1" || value == "true");
            else if (key == "compute_metrics") config.compute_metrics = (value == "1" || value == "true");
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error parsing parameter '" << key << "': " << e.what() << std::endl;
        }
    }

    file.close();
    return true;
}

// ============================================================================
// Load the pair configuration file. Expected format per block:
//
//   <source_id>-<target_id>
//   r00 r01 r02 tx
//   r10 r11 r12 ty
//   r20 r21 r22 tz
//   0   0   0   1
//   <blank line>
// ============================================================================
std::vector<PointCloudPair> loadPairsConfig(const std::string &config_file)
{
    std::vector<PointCloudPair> pairs;
    std::ifstream file(config_file);

    if (!file.is_open())
    {
        std::cerr << "Failed to open pairs config file: " << config_file << std::endl;
        return pairs;
    }

    std::string line;
    PointCloudPair current_pair;
    int matrix_row = 0;
    bool reading_pair = false;

    while (std::getline(file, line))
    {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty())
        {
            if (reading_pair && matrix_row == 4)
            {
                pairs.push_back(current_pair);
                reading_pair = false;
                matrix_row = 0;
            }
            continue;
        }

        // Detect pair identifier like "182-183" or "09-08"
        if (line.find('-') != std::string::npos && !reading_pair)
        {
            size_t dash_pos = line.find('-');
            current_pair.source_id = line.substr(0, dash_pos);
            current_pair.target_id = line.substr(dash_pos + 1);
            current_pair.gt_matrix = Eigen::Matrix4f::Identity();
            reading_pair = true;
            matrix_row = 0;
            continue;
        }

        if (reading_pair && matrix_row < 4)
        {
            std::istringstream iss(line);
            for (int col = 0; col < 4; ++col)
            {
                if (!(iss >> current_pair.gt_matrix(matrix_row, col)))
                {
                    std::cerr << "Error parsing matrix row " << matrix_row << std::endl;
                    break;
                }
            }
            matrix_row++;
        }
    }

    // Handle the last pair if the file does not end with a blank line
    if (reading_pair && matrix_row == 4)
    {
        pairs.push_back(current_pair);
    }

    file.close();
    return pairs;
}

// ============================================================================
// Estimate a sensible feature search radius from point cloud density
// (median of average k-NN distances, scaled by a multiplier).
// ============================================================================
double estimateSearchRadius(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud,
                            float multiplier = 7.5f, int k_neighbors = 10, int sample_size = 1000)
{
    if (cloud->empty())
    {
        return 0.3;  // sensible default
    }

    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
    tree->setInputCloud(cloud);

    int actual_sample_size = std::min(sample_size, static_cast<int>(cloud->size()));

    std::vector<int> sample_indices(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i)
        sample_indices[i] = static_cast<int>(i);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(sample_indices.begin(), sample_indices.end(), g);
    sample_indices.resize(actual_sample_size);

    std::vector<double> avg_distances;
    avg_distances.reserve(actual_sample_size);

    for (int idx : sample_indices)
    {
        std::vector<int> nn_indices(k_neighbors + 1);
        std::vector<float> nn_distances(k_neighbors + 1);

        if (tree->nearestKSearch(cloud->points[idx], k_neighbors + 1, nn_indices, nn_distances) > 0)
        {
            double sum = 0.0;
            // Skip the self-match at index 0
            for (size_t i = 1; i < nn_distances.size(); ++i)
            {
                sum += std::sqrt(nn_distances[i]);
            }
            avg_distances.push_back(sum / k_neighbors);
        }
    }

    if (avg_distances.empty())
    {
        return 0.3;
    }

    std::sort(avg_distances.begin(), avg_distances.end());
    double median_distance = avg_distances[avg_distances.size() / 2];

    double recommended_radius = median_distance * multiplier;
    recommended_radius = std::max(0.05, std::min(recommended_radius, 2.0));

    return recommended_radius;
}


// ============================================================================
// Metric implementations
// ============================================================================

/**
 * Relative Rotation Error (degrees).
 *   RRE = arccos( clip((trace(R_est * R_gt^T) - 1) / 2, -1, 1) ) * 180 / pi.
 *
 * Note on the formula: R_est * R_gt^T and R_est^T * R_gt have the same trace
 * (one is the transpose of the other for rotation matrices), so either form
 * gives the same angle. We use R_est * R_gt^T here to follow the literature
 * convention and to match the Python evaluators in this repository.
 *
 * The trace itself lives in [-1, 3]; we clip it explicitly for numerical
 * safety, then clip (trace-1)/2 to the valid acos domain [-1, 1].
 */
double computeRRE(const Eigen::Matrix4f &estimated, const Eigen::Matrix4f &ground_truth)
{
    Eigen::Matrix3f R_est = estimated.block<3, 3>(0, 0);
    Eigen::Matrix3f R_gt  = ground_truth.block<3, 3>(0, 0);

    Eigen::Matrix3f R_diff = R_est * R_gt.transpose();

    float trace = R_diff.trace();
    trace = std::max(-1.0f, std::min(3.0f, trace));

    float cos_angle = (trace - 1.0f) / 2.0f;
    cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));

    float angle_rad = std::acos(cos_angle);
    float angle_deg = angle_rad * 180.0f / static_cast<float>(M_PI);

    return static_cast<double>(angle_deg);
}

/**
 * Relative Translation Error (meters).
 *   RTE = || t_est - t_gt ||_2
 */
double computeRTE(const Eigen::Matrix4f &estimated, const Eigen::Matrix4f &ground_truth)
{
    Eigen::Vector3f t_est = estimated.block<3, 1>(0, 3);
    Eigen::Vector3f t_gt  = ground_truth.block<3, 1>(0, 3);

    return static_cast<double>((t_est - t_gt).norm());
}

/**
 * Inlier ratio of feature correspondences, validated by the ground-truth
 * transform: a correspondence (p_s, p_t) is an inlier iff
 *     || T_gt * p_s - p_t || < distance_threshold.
 *
 * Returns: (inlier_ratio, num_correspondences, num_inliers).
 */
template<typename FeatureT>
std::tuple<double, int, int> computeInlierRatio(
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &source,
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &target,
    const typename pcl::PointCloud<FeatureT>::Ptr &source_features,
    const typename pcl::PointCloud<FeatureT>::Ptr &target_features,
    const Eigen::Matrix4f &ground_truth,
    float distance_threshold)
{
    pcl::search::KdTree<FeatureT> feature_tree;
    feature_tree.setInputCloud(target_features);

    int num_correspondences = 0;
    int num_inliers = 0;

    const int num_features = static_cast<int>(source_features->size());

    #pragma omp parallel reduction(+:num_correspondences, num_inliers)
    {
        std::vector<int> indices(1);
        std::vector<float> distances(1);

        #pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < num_features; ++i)
        {
            if (feature_tree.nearestKSearch(source_features->points[i], 1, indices, distances) > 0)
            {
                num_correspondences++;

                Eigen::Vector4f src_pt(source->points[i].x, source->points[i].y, source->points[i].z, 1.0f);
                Eigen::Vector4f transformed_pt = ground_truth * src_pt;

                const auto &tgt_pt = target->points[indices[0]];
                float dx = transformed_pt[0] - tgt_pt.x;
                float dy = transformed_pt[1] - tgt_pt.y;
                float dz = transformed_pt[2] - tgt_pt.z;
                float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                if (dist < distance_threshold)
                {
                    num_inliers++;
                }
            }
        }
    }

    double ir = (num_correspondences > 0) ?
                static_cast<double>(num_inliers) / num_correspondences : 0.0;

    return std::make_tuple(ir, num_correspondences, num_inliers);
}

/**
 * Same as computeInlierRatio, but also returns the per-correspondence
 * distances so that the inlier ratio can be recomputed for other
 * distance thresholds without re-running the feature matching.
 *
 * Returns: (inlier_ratio, num_correspondences, num_inliers, distances).
 */
template<typename FeatureT>
std::tuple<double, int, int, std::vector<float>> computeInlierRatioWithDistances(
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &source,
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &target,
    const typename pcl::PointCloud<FeatureT>::Ptr &source_features,
    const typename pcl::PointCloud<FeatureT>::Ptr &target_features,
    const Eigen::Matrix4f &ground_truth,
    float distance_threshold)
{
    pcl::search::KdTree<FeatureT> feature_tree;
    feature_tree.setInputCloud(target_features);

    const int num_features = static_cast<int>(source_features->size());
    std::vector<float> all_distances(num_features, -1.0f);

    int num_correspondences = 0;
    int num_inliers = 0;

    // Serial loop to preserve per-index distances.
    std::vector<int> indices(1);
    std::vector<float> distances(1);

    for (int i = 0; i < num_features; ++i)
    {
        if (feature_tree.nearestKSearch(source_features->points[i], 1, indices, distances) > 0)
        {
            num_correspondences++;

            Eigen::Vector4f src_pt(source->points[i].x, source->points[i].y, source->points[i].z, 1.0f);
            Eigen::Vector4f transformed_pt = ground_truth * src_pt;

            const auto &tgt_pt = target->points[indices[0]];
            float dx = transformed_pt[0] - tgt_pt.x;
            float dy = transformed_pt[1] - tgt_pt.y;
            float dz = transformed_pt[2] - tgt_pt.z;
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

            all_distances[i] = dist;

            if (dist < distance_threshold)
            {
                num_inliers++;
            }
        }
    }

    std::vector<float> valid_distances;
    valid_distances.reserve(num_correspondences);
    for (int i = 0; i < num_features; ++i)
    {
        if (all_distances[i] >= 0.0f)
        {
            valid_distances.push_back(all_distances[i]);
        }
    }

    double ir = (num_correspondences > 0) ?
                static_cast<double>(num_inliers) / num_correspondences : 0.0;

    return std::make_tuple(ir, num_correspondences, num_inliers, valid_distances);
}

/**
 * Given the per-correspondence distances, recompute the inlier ratio for a
 * given distance threshold.
 */
double computeIRFromDistances(const std::vector<float> &distances, float threshold)
{
    if (distances.empty()) return 0.0;

    int inliers = 0;
    for (float d : distances)
    {
        if (d < threshold) inliers++;
    }
    return static_cast<double>(inliers) / distances.size();
}

/**
 * Jointly compute:
 *   - Registration RMSE = sqrt( mean( || T_est * p_i - T_gt * p_i ||^2 ) ).
 *     This is the transform-difference RMSE evaluated on the source cloud,
 *     which is a fair proxy when the source is densely sampled.
 *   - Registration IR  = fraction of transformed source points whose nearest
 *     neighbour in the target cloud lies within `distance_threshold`.
 *
 * Both metrics are computed in a single pass over the source cloud and one
 * shared KD-tree over the target cloud.
 *
 * Returns pair<RMSE, Registration_IR>. A value of -1 indicates "not computed".
 */
std::pair<double, double> computeRegistrationMetrics(
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &source,
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &target,
    const Eigen::Matrix4f &estimated,
    const Eigen::Matrix4f &ground_truth,
    float distance_threshold)
{
    if (source->empty()) return std::make_pair(-1.0, -1.0);

    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
    if (!target->empty())
    {
        tree->setInputCloud(target);
    }

    double sum_sq_rmse = 0.0;
    int valid_count = 0;
    int inlier_count = 0;

    std::vector<int> indices(1);
    std::vector<float> distances(1);

    for (size_t i = 0; i < source->size(); ++i)
    {
        const auto &pt = source->points[i];

        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;

        valid_count++;

        Eigen::Vector4f p(pt.x, pt.y, pt.z, 1.0f);
        Eigen::Vector4f p_est = estimated * p;
        Eigen::Vector4f p_gt  = ground_truth * p;

        float dx = p_est[0] - p_gt[0];
        float dy = p_est[1] - p_gt[1];
        float dz = p_est[2] - p_gt[2];
        sum_sq_rmse += static_cast<double>(dx*dx + dy*dy + dz*dz);

        if (!target->empty())
        {
            pcl::PointXYZRGBA search_pt;
            search_pt.x = p_est[0];
            search_pt.y = p_est[1];
            search_pt.z = p_est[2];

            if (tree->nearestKSearch(search_pt, 1, indices, distances) > 0)
            {
                if (std::sqrt(distances[0]) < distance_threshold)
                {
                    inlier_count++;
                }
            }
        }
    }

    if (valid_count == 0) return std::make_pair(-1.0, -1.0);

    double rmse   = std::sqrt(sum_sq_rmse / valid_count);
    double reg_ir = target->empty() ? -1.0 : static_cast<double>(inlier_count) / valid_count;

    return std::make_pair(rmse, reg_ir);
}

// ============================================================================
// Voxel-grid filter
// ============================================================================
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr voxelGridFilter(
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud,
    float leaf_size)
{
    if (leaf_size <= 0.0f) return cloud;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::VoxelGrid<pcl::PointXYZRGBA> filter;
    filter.setInputCloud(cloud);
    filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    filter.filter(*filtered);
    return filtered;
}

// ============================================================================
// Surface normal estimation (k-NN, parallel)
// ============================================================================
pcl::PointCloud<pcl::Normal>::Ptr estimateNormals(
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud,
    int k_search)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimationOMP<pcl::PointXYZRGBA, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);

    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(k_search);
    ne.compute(*normals);

    return normals;
}

// ============================================================================
// FPFH descriptor (parallel)
// ============================================================================
pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFH(
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr &normals,
    float radius)
{
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::FPFHEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh;
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);

    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(radius);
    fpfh.compute(*features);

    return features;
}

// ============================================================================
// CSHOT (Color-SHOT) descriptor (parallel)
// ============================================================================
pcl::PointCloud<pcl::SHOT1344>::Ptr computeCSHOT(
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud,
    const pcl::PointCloud<pcl::Normal>::Ptr &normals,
    float radius)
{
    pcl::PointCloud<pcl::SHOT1344>::Ptr features(new pcl::PointCloud<pcl::SHOT1344>);
    pcl::SHOTColorEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::SHOT1344> shot;
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);

    shot.setInputCloud(cloud);
    shot.setInputNormals(normals);
    shot.setSearchMethod(tree);
    shot.setRadiusSearch(radius);
    shot.compute(*features);

    return features;
}

// ============================================================================
// Visualize a single registration result
// ============================================================================
void visualizeSingleResult(
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &source,
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &target,
    const Eigen::Matrix4f &transformation,
    const std::string &method_name)
{
    std::cout << "  Visualizing " << method_name << " result..." << std::endl;
    std::cout << "  (Press 'q' to close and continue)" << std::endl;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::transformPointCloud(*source, *aligned, transformation);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(method_name + " Result"));
    viewer->setBackgroundColor(0.05, 0.05, 0.05);

    // Target cloud - green
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> target_color(target, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZRGBA>(target, target_color, "target");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target");

    // Aligned source cloud - blue
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> aligned_color(aligned, 0, 100, 255);
    viewer->addPointCloud<pcl::PointXYZRGBA>(aligned, aligned_color, "aligned");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "aligned");

    viewer->addCoordinateSystem(1.0);
    viewer->addText(method_name + " Result (Green: Target, Blue: Aligned Source)",
                    10, 30, 16, 1.0, 1.0, 1.0, "title");
    viewer->addText("Press 'q' to continue", 10, 10, 12, 0.8, 0.8, 0.8, "hint");

    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 50, 0, 0, 0, 0, 1, 0);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }

    viewer->close();
}

// ============================================================================
// FPFH + RANSAC
// ============================================================================
EvalResult evaluateFPFH(
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &source,
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &target,
    const pcl::PointCloud<pcl::Normal>::Ptr &source_normals,
    const pcl::PointCloud<pcl::Normal>::Ptr &target_normals,
    const Eigen::Matrix4f &ground_truth,
    const EvalConfig &config,
    bool show_visualization = true)
{
    EvalResult result;
    result.method_name = "FPFH + RANSAC";

    std::cout << "\n========== Evaluating FPFH + RANSAC ==========" << std::endl;

    // Time only feature computation + alignment.
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Computing FPFH features..." << std::endl;
    auto source_features = computeFPFH(source, source_normals, config.feature_radius);
    auto target_features = computeFPFH(target, target_normals, config.feature_radius);

    std::cout << "  Source features: " << source_features->size() << std::endl;
    std::cout << "  Target features: " << target_features->size() << std::endl;

    std::cout << "Running RANSAC alignment..." << std::endl;
    pcl::SampleConsensusPrerejective<pcl::PointXYZRGBA, pcl::PointXYZRGBA, pcl::FPFHSignature33> align;
    align.setInputSource(source);
    align.setInputTarget(target);
    align.setSourceFeatures(source_features);
    align.setTargetFeatures(target_features);
    align.setMaximumIterations(config.ransac_max_iterations);
    align.setNumberOfSamples(config.ransac_nr_samples);
    align.setCorrespondenceRandomness(5);
    align.setSimilarityThreshold(config.ransac_similarity_threshold);
    align.setMaxCorrespondenceDistance(config.ransac_max_correspondence_distance);
    align.setInlierFraction(config.ransac_inlier_fraction);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZRGBA>);
    align.align(*aligned);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // IR computation does not count toward timing.
    if (config.compute_inlier_ratio)
    {
        std::cout << "Computing inlier ratio (not included in timing)..." << std::endl;
        auto ir_result = computeInlierRatioWithDistances<pcl::FPFHSignature33>(
            source, target, source_features, target_features,
            ground_truth, config.inlier_distance_threshold);
        double ir = std::get<0>(ir_result);
        int num_corr = std::get<1>(ir_result);
        int num_inl  = std::get<2>(ir_result);
        result.correspondence_distances = std::get<3>(ir_result);
        result.inlier_ratio = ir;
        result.num_correspondences = num_corr;
        result.num_inliers = num_inl;

        std::cout << "  Feature correspondences: " << num_corr << std::endl;
        std::cout << "  Inlier ratio (IR): " << std::fixed << std::setprecision(2)
                  << (ir * 100.0) << "%" << std::endl;
    }
    else
    {
        result.inlier_ratio = -1.0;
        result.num_correspondences = 0;
        result.num_inliers = 0;
    }

    if (align.hasConverged())
    {
        result.success = true;
        result.transformation = align.getFinalTransformation();

        if (config.compute_metrics)
        {
            result.rre = computeRRE(result.transformation, ground_truth);
            result.rte = computeRTE(result.transformation, ground_truth);

            auto metrics = computeRegistrationMetrics(source, target, result.transformation,
                                                      ground_truth, config.inlier_distance_threshold);
            result.inlier_rmse = metrics.first;
            result.registration_ir = metrics.second;

            std::cout << "  Alignment converged!" << std::endl;
            if (result.inlier_rmse >= 0)
            {
                std::cout << "  Registration RMSE: " << std::fixed << std::setprecision(4)
                          << result.inlier_rmse << " m" << std::endl;
            }
            if (result.registration_ir >= 0)
            {
                std::cout << "  Registration IR: " << std::fixed << std::setprecision(2)
                          << (result.registration_ir * 100.0) << "%" << std::endl;
            }
        }
        else
        {
            std::cout << "  Alignment converged! (metrics computation disabled)" << std::endl;
        }

        if (show_visualization)
        {
            visualizeSingleResult(source, target, result.transformation, result.method_name);
        }
    }
    else
    {
        result.success = false;
        result.inlier_rmse = -1.0;
        result.registration_ir = -1.0;
        std::cout << "  Alignment failed to converge!" << std::endl;
    }

    std::cout << "  Time: " << std::fixed << std::setprecision(2) << result.time_ms << " ms" << std::endl;
    std::cout << "==============================================" << std::endl;

    return result;
}

// ============================================================================
// CSHOT + RANSAC
// ============================================================================
EvalResult evaluateCSHOT(
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &source,
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &target,
    const pcl::PointCloud<pcl::Normal>::Ptr &source_normals,
    const pcl::PointCloud<pcl::Normal>::Ptr &target_normals,
    const Eigen::Matrix4f &ground_truth,
    const EvalConfig &config,
    bool show_visualization = true)
{
    EvalResult result;
    result.method_name = "CSHOT + RANSAC";

    std::cout << "\n========== Evaluating CSHOT + RANSAC ==========" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Computing CSHOT features..." << std::endl;
    auto source_features = computeCSHOT(source, source_normals, config.feature_radius);
    auto target_features = computeCSHOT(target, target_normals, config.feature_radius);

    // Drop points whose descriptors contain NaNs.
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_valid(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_valid(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::SHOT1344>::Ptr source_features_valid(new pcl::PointCloud<pcl::SHOT1344>);
    pcl::PointCloud<pcl::SHOT1344>::Ptr target_features_valid(new pcl::PointCloud<pcl::SHOT1344>);

    for (size_t i = 0; i < source_features->size(); ++i)
    {
        bool valid = true;
        for (int j = 0; j < 1344 && valid; ++j)
        {
            if (!std::isfinite(source_features->points[i].descriptor[j]))
                valid = false;
        }
        if (valid)
        {
            source_valid->push_back(source->points[i]);
            source_features_valid->push_back(source_features->points[i]);
        }
    }

    for (size_t i = 0; i < target_features->size(); ++i)
    {
        bool valid = true;
        for (int j = 0; j < 1344 && valid; ++j)
        {
            if (!std::isfinite(target_features->points[i].descriptor[j]))
                valid = false;
        }
        if (valid)
        {
            target_valid->push_back(target->points[i]);
            target_features_valid->push_back(target_features->points[i]);
        }
    }

    std::cout << "  Source features: " << source_features_valid->size()
              << " (valid) / " << source_features->size() << " (total)" << std::endl;
    std::cout << "  Target features: " << target_features_valid->size()
              << " (valid) / " << target_features->size() << " (total)" << std::endl;

    std::cout << "Running RANSAC alignment..." << std::endl;
    pcl::SampleConsensusPrerejective<pcl::PointXYZRGBA, pcl::PointXYZRGBA, pcl::SHOT1344> align;
    align.setInputSource(source_valid);
    align.setInputTarget(target_valid);
    align.setSourceFeatures(source_features_valid);
    align.setTargetFeatures(target_features_valid);
    align.setMaximumIterations(config.ransac_max_iterations);
    align.setNumberOfSamples(config.ransac_nr_samples);
    align.setCorrespondenceRandomness(5);
    align.setSimilarityThreshold(config.ransac_similarity_threshold);
    align.setMaxCorrespondenceDistance(config.ransac_max_correspondence_distance);
    align.setInlierFraction(config.ransac_inlier_fraction);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZRGBA>);
    align.align(*aligned);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (config.compute_inlier_ratio)
    {
        std::cout << "Computing inlier ratio (not included in timing)..." << std::endl;
        auto ir_result = computeInlierRatioWithDistances<pcl::SHOT1344>(
            source_valid, target_valid, source_features_valid, target_features_valid,
            ground_truth, config.inlier_distance_threshold);
        double ir = std::get<0>(ir_result);
        int num_corr = std::get<1>(ir_result);
        int num_inl  = std::get<2>(ir_result);
        result.correspondence_distances = std::get<3>(ir_result);
        result.inlier_ratio = ir;
        result.num_correspondences = num_corr;
        result.num_inliers = num_inl;

        std::cout << "  Feature correspondences: " << num_corr << std::endl;
        std::cout << "  Inlier ratio (IR): " << std::fixed << std::setprecision(2)
                  << (ir * 100.0) << "%" << std::endl;
    }
    else
    {
        result.inlier_ratio = -1.0;
        result.num_correspondences = 0;
        result.num_inliers = 0;
    }

    if (align.hasConverged())
    {
        result.success = true;
        result.transformation = align.getFinalTransformation();

        if (config.compute_metrics)
        {
            result.rre = computeRRE(result.transformation, ground_truth);
            result.rte = computeRTE(result.transformation, ground_truth);

            auto metrics = computeRegistrationMetrics(source, target, result.transformation,
                                                      ground_truth, config.inlier_distance_threshold);
            result.inlier_rmse = metrics.first;
            result.registration_ir = metrics.second;

            std::cout << "  Alignment converged!" << std::endl;
            if (result.inlier_rmse >= 0)
            {
                std::cout << "  Registration RMSE: " << std::fixed << std::setprecision(4)
                          << result.inlier_rmse << " m" << std::endl;
            }
            if (result.registration_ir >= 0)
            {
                std::cout << "  Registration IR: " << std::fixed << std::setprecision(2)
                          << (result.registration_ir * 100.0) << "%" << std::endl;
            }
        }
        else
        {
            std::cout << "  Alignment converged! (metrics computation disabled)" << std::endl;
        }

        if (show_visualization)
        {
            visualizeSingleResult(source, target, result.transformation, result.method_name);
        }
    }
    else
    {
        result.success = false;
        result.inlier_rmse = -1.0;
        result.registration_ir = -1.0;
        std::cout << "  Alignment failed to converge!" << std::endl;
    }

    std::cout << "  Time: " << std::fixed << std::setprecision(2) << result.time_ms << " ms" << std::endl;
    std::cout << "===============================================" << std::endl;

    return result;
}

// ============================================================================
// ICP
// ============================================================================
EvalResult evaluateICP(
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &source,
    const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &target,
    const Eigen::Matrix4f &ground_truth,
    const EvalConfig &config,
    const Eigen::Matrix4f &initial_guess = Eigen::Matrix4f::Identity(),
    bool show_visualization = true)
{
    EvalResult result;
    result.method_name = "ICP";

    std::cout << "\n========== Evaluating ICP ==========" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Running ICP alignment..." << std::endl;
    pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaxCorrespondenceDistance(config.icp_max_correspondence_distance);
    icp.setMaximumIterations(config.icp_max_iterations);
    icp.setTransformationEpsilon(config.icp_transformation_epsilon);
    icp.setEuclideanFitnessEpsilon(config.icp_euclidean_fitness_epsilon);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZRGBA>);
    icp.align(*aligned, initial_guess);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (icp.hasConverged())
    {
        result.success = true;
        result.transformation = icp.getFinalTransformation();

        if (config.compute_metrics)
        {
            result.rre = computeRRE(result.transformation, ground_truth);
            result.rte = computeRTE(result.transformation, ground_truth);

            auto metrics = computeRegistrationMetrics(source, target, result.transformation,
                                                      ground_truth, config.inlier_distance_threshold);
            result.inlier_rmse = metrics.first;
            result.registration_ir = metrics.second;

            std::cout << "  ICP converged!" << std::endl;
            std::cout << "  Fitness score: " << icp.getFitnessScore() << std::endl;
            if (result.inlier_rmse >= 0)
            {
                std::cout << "  Registration RMSE: " << std::fixed << std::setprecision(4)
                          << result.inlier_rmse << " m" << std::endl;
            }
            if (result.registration_ir >= 0)
            {
                std::cout << "  Registration IR: " << std::fixed << std::setprecision(2)
                          << (result.registration_ir * 100.0) << "%" << std::endl;
            }
        }
        else
        {
            std::cout << "  ICP converged! (metrics computation disabled)" << std::endl;
            std::cout << "  Fitness score: " << icp.getFitnessScore() << std::endl;
        }

        if (show_visualization)
        {
            visualizeSingleResult(source, target, result.transformation, result.method_name);
        }
    }
    else
    {
        result.success = false;
        result.inlier_rmse = -1.0;
        result.registration_ir = -1.0;
        std::cout << "  ICP failed to converge!" << std::endl;
    }

    // ICP has no feature-matching IR.
    result.inlier_ratio = -1.0;
    result.num_correspondences = 0;
    result.num_inliers = 0;

    std::cout << "  Time: " << std::fixed << std::setprecision(2) << result.time_ms << " ms" << std::endl;
    std::cout << "======================================" << std::endl;

    return result;
}

// ============================================================================
// Aggregate result printers
// ============================================================================
void printBatchResults(const std::map<std::string, MethodStats> &stats_map)
{
    std::cout << "\n" << std::string(150, '=') << std::endl;
    std::cout << "                                 BATCH EVALUATION RESULTS SUMMARY" << std::endl;
    std::cout << std::string(150, '=') << std::endl;

    std::cout << "\n" << std::string(150, '-') << std::endl;
    std::cout << std::left << std::setw(18) << "Method"
              << std::right << std::setw(18) << "RRE(deg)"
              << std::setw(18) << "RTE(m)"
              << std::setw(20) << "RMSE(m)"
              << std::setw(18) << "Reg IR(%)"
              << std::setw(10) << "FMR(%)"
              << std::setw(10) << "RR(%)"
              << std::setw(14) << "Time(s/KPts)"
              << std::endl;
    std::cout << std::string(150, '-') << std::endl;

    for (const auto &pair : stats_map)
    {
        const MethodStats &s = pair.second;

        std::ostringstream rre_str, rte_str, rmse_str, reg_ir_str;

        rre_str << std::fixed << std::setprecision(2) << s.avg_rre
                << " +- " << std::setprecision(2) << s.std_rre;
        rte_str << std::fixed << std::setprecision(2) << s.avg_rte
                << " +- " << std::setprecision(2) << s.std_rte;

        if (s.avg_inlier_rmse >= 0)
        {
            rmse_str << std::fixed << std::setprecision(3) << s.avg_inlier_rmse
                     << " +- " << std::setprecision(3) << s.std_inlier_rmse;
        }
        else
        {
            rmse_str << "N/A";
        }

        if (s.avg_registration_ir >= 0)
        {
            reg_ir_str << std::fixed << std::setprecision(1) << (s.avg_registration_ir * 100.0)
                       << " +- " << std::setprecision(1) << (s.std_registration_ir * 100.0);
        }
        else
        {
            reg_ir_str << "N/A";
        }

        std::cout << std::left << std::setw(18) << s.method_name
                  << std::right << std::setw(18) << rre_str.str()
                  << std::setw(18) << rte_str.str()
                  << std::setw(20) << rmse_str.str()
                  << std::setw(18) << reg_ir_str.str()
                  << std::setw(10) << std::fixed << std::setprecision(1) << (s.fmr * 100.0)
                  << std::setw(10) << std::fixed << std::setprecision(1) << (s.rr * 100.0)
                  << std::setw(14) << std::fixed << std::setprecision(4) << s.time_per_kpts
                  << std::endl;
    }

    std::cout << std::string(150, '-') << std::endl;
    std::cout << "\nNote:" << std::endl;
    std::cout << "  RRE, RTE, RMSE and Reg IR show mean +- std (standard deviation)" << std::endl;
    std::cout << "  RMSE: sqrt(mean(||T_est*p - T_gt*p||^2)), based on transformation difference" << std::endl;
    std::cout << "  Reg IR: Registration Inlier Ratio, overlap between transformed source and target" << std::endl;
    std::cout << "  FMR (Feature Matching Recall): IR > threshold is considered correct matching" << std::endl;
    std::cout << "  RR  (Registration Recall):    RTE < threshold AND RRE < threshold" << std::endl;
    std::cout << "  Time: seconds per 1000 points (s/KPts)" << std::endl;
    std::cout << "  Total pairs evaluated: " << (stats_map.empty() ? 0 : stats_map.begin()->second.total_pairs) << std::endl;
    std::cout << std::string(150, '=') << std::endl;
}

// ============================================================================
// FMR as a function of IR threshold (distance threshold fixed)
// ============================================================================
void printFMRByIRThreshold(
    const std::map<std::string, std::vector<EvalResult>> &all_results,
    const std::vector<float> &ir_thresholds)
{
    std::cout << "\n" << std::string(110, '=') << std::endl;
    std::cout << "           FMR vs Inlier Ratio Threshold (fixed distance threshold)" << std::endl;
    std::cout << std::string(110, '=') << std::endl;

    std::cout << "\n" << std::left << std::setw(18) << "Method";
    for (float th : ir_thresholds)
    {
        std::ostringstream oss;
        double pct = th * 100;
        if (pct < 1.0)
            oss << "IR>" << std::fixed << std::setprecision(2) << pct << "%";
        else
            oss << "IR>" << std::fixed << std::setprecision(1) << pct << "%";
        std::cout << std::right << std::setw(10) << oss.str();
    }
    std::cout << std::endl;
    std::cout << std::string(110, '-') << std::endl;

    for (const auto &pair : all_results)
    {
        const std::string &method = pair.first;
        const std::vector<EvalResult> &results = pair.second;

        // ICP has no feature matching, skip
        if (method == "ICP") continue;

        std::cout << std::left << std::setw(18) << method;

        for (float th : ir_thresholds)
        {
            int fmr_count = 0;
            int valid_count = 0;

            for (const EvalResult &r : results)
            {
                if (r.inlier_ratio >= 0)
                {
                    valid_count++;
                    if (r.inlier_ratio > th)
                    {
                        fmr_count++;
                    }
                }
            }

            double fmr = (valid_count > 0) ?
                         static_cast<double>(fmr_count) / valid_count * 100.0 : 0.0;
            std::cout << std::right << std::setw(10) << std::fixed
                      << std::setprecision(1) << fmr;
        }
        std::cout << std::endl;
    }

    std::cout << std::string(110, '-') << std::endl;
    std::cout << "Note: FMR(%) = percentage of pairs with IR > threshold" << std::endl;
    std::cout << std::string(110, '=') << std::endl;
}

// ============================================================================
// FMR as a function of inlier distance threshold (IR threshold fixed)
// ============================================================================
void printFMRByDistanceThreshold(
    const std::map<std::string, std::vector<EvalResult>> &all_results,
    const std::vector<float> &distance_thresholds,
    float ir_threshold)
{
    std::cout << "\n" << std::string(110, '=') << std::endl;
    std::cout << "           FMR vs Inlier Distance Threshold (IR threshold = "
              << std::fixed << std::setprecision(1) << (ir_threshold * 100) << "%)" << std::endl;
    std::cout << std::string(110, '=') << std::endl;

    std::cout << "\n" << std::left << std::setw(18) << "Method";
    for (float th : distance_thresholds)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << th << "m";
        std::cout << std::right << std::setw(10) << oss.str();
    }
    std::cout << std::endl;
    std::cout << std::string(110, '-') << std::endl;

    for (const auto &pair : all_results)
    {
        const std::string &method = pair.first;
        const std::vector<EvalResult> &results = pair.second;

        if (method == "ICP") continue;

        std::cout << std::left << std::setw(18) << method;

        for (float dist_th : distance_thresholds)
        {
            int fmr_count = 0;
            int valid_count = 0;

            for (const EvalResult &r : results)
            {
                if (!r.correspondence_distances.empty())
                {
                    valid_count++;
                    double ir = computeIRFromDistances(r.correspondence_distances, dist_th);
                    if (ir > ir_threshold)
                    {
                        fmr_count++;
                    }
                }
            }

            double fmr = (valid_count > 0) ?
                         static_cast<double>(fmr_count) / valid_count * 100.0 : 0.0;
            std::cout << std::right << std::setw(10) << std::fixed
                      << std::setprecision(1) << fmr;
        }
        std::cout << std::endl;
    }

    std::cout << std::string(110, '-') << std::endl;
    std::cout << "Note: IR is recalculated for each distance threshold using saved correspondence distances" << std::endl;
    std::cout << std::string(110, '=') << std::endl;
}

// ============================================================================
// Save per-method estimated transforms (same format as the GT pairs file).
// ============================================================================
void saveRegistrationResults(
    const std::map<std::string, std::vector<EvalResult>> &all_results,
    const std::string &output_path,
    const std::string &method_name)
{
    std::ofstream file(output_path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return;
    }

    auto it = all_results.find(method_name);
    if (it == all_results.end())
    {
        std::cerr << "Method not found: " << method_name << std::endl;
        file.close();
        return;
    }

    const std::vector<EvalResult> &results = it->second;

    for (const EvalResult &r : results)
    {
        if (!r.success) continue;

        file << r.pair_id << std::endl;

        const Eigen::Matrix4f &T = r.transformation;
        file << std::fixed << std::setprecision(6);
        for (int row = 0; row < 4; ++row)
        {
            file << T(row, 0) << " " << T(row, 1) << " " << T(row, 2) << " " << T(row, 3) << std::endl;
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Saved registration results to: " << output_path << std::endl;
}

void saveAllRegistrationResults(
    const std::map<std::string, std::vector<EvalResult>> &all_results,
    const std::string &output_directory)
{
    for (const auto &pair : all_results)
    {
        const std::string &method = pair.first;

        std::string filename = method;
        std::replace(filename.begin(), filename.end(), ' ', '_');
        std::replace(filename.begin(), filename.end(), '+', '_');
        filename += "_results.txt";

        std::string output_path = output_directory.empty() ?
                                  filename : (output_directory + "/" + filename);

        saveRegistrationResults(all_results, output_path, method);
    }
}

// ============================================================================
// Entry point
// ============================================================================
int main(int argc, char **argv)
{
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "    Point Cloud Registration Batch Evaluation Program" << std::endl;
    std::cout << "    Methods: FPFH + RANSAC, CSHOT + RANSAC, ICP" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

    EvalConfig config;
    std::string config_file = "eval_config.txt";

    if (argc > 1)
    {
        config_file = argv[1];
    }

    std::cout << "\nLoading configuration from: " << config_file << std::endl;
    if (!loadConfig(config_file, config))
    {
        std::cerr << "Failed to load config file!" << std::endl;
        return -1;
    }

    // Use the directory of the config file as the base for relative paths.
    std::string config_dir = "";
    size_t last_slash = config_file.find_last_of("/\\");
    if (last_slash != std::string::npos)
    {
        config_dir = config_file.substr(0, last_slash + 1);
    }

    if (config.data_directory.empty() || config.data_directory == ".")
    {
        config.data_directory = config_dir.empty() ? "." : config_dir.substr(0, config_dir.length() - 1);
    }
    else if (config.data_directory[0] != '/' && config.data_directory[0] != '\\' &&
             (config.data_directory.length() < 2 || config.data_directory[1] != ':'))
    {
        // Relative path -> resolve against config-file directory
        config.data_directory = config_dir + config.data_directory;
    }

    std::cout << "Data directory: " << config.data_directory << std::endl;

    if (config.data_directory.empty())
    {
        std::cerr << "Error: data_directory not specified!" << std::endl;
        return -1;
    }

    std::string pairs_config_path = config.data_directory + "/" + config.pairs_config_file;
    std::cout << "Loading pairs config from: " << pairs_config_path << std::endl;

    std::vector<PointCloudPair> pairs = loadPairsConfig(pairs_config_path);
    if (pairs.empty())
    {
        std::cerr << "Error: No point cloud pairs found!" << std::endl;
        return -1;
    }

    std::cout << "Found " << pairs.size() << " point cloud pairs to evaluate.\n" << std::endl;

    std::map<std::string, MethodStats> stats_map;
    stats_map["FPFH + RANSAC"] = MethodStats();
    stats_map["FPFH + RANSAC"].method_name = "FPFH + RANSAC";
    stats_map["CSHOT + RANSAC"] = MethodStats();
    stats_map["CSHOT + RANSAC"].method_name = "CSHOT + RANSAC";
    stats_map["ICP"] = MethodStats();
    stats_map["ICP"].method_name = "ICP";

    std::map<std::string, std::vector<EvalResult>> all_results;

    for (size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx)
    {
        const PointCloudPair &pair = pairs[pair_idx];
        std::string pair_id = pair.source_id + "-" + pair.target_id;

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "[" << (pair_idx + 1) << "/" << pairs.size() << "] Evaluating pair: "
                  << pair_id << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        // Build file paths (numeric IDs are zero-padded to 3 digits, e.g. 9 -> 009).
        auto formatId = [](const std::string &id) -> std::string {
            try {
                int num = std::stoi(id);
                std::ostringstream oss;
                oss << std::setw(3) << std::setfill('0') << num;
                return oss.str();
            } catch (...) {
                return id;
            }
        };
        std::string source_path = config.data_directory + "/Scan_" + formatId(pair.source_id) + ".pcd";
        std::string target_path = config.data_directory + "/Scan_" + formatId(pair.target_id) + ".pcd";

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

        std::cout << "Loading source: " << source_path << std::endl;
        if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(source_path, *source_cloud) == -1)
        {
            std::cerr << "  Failed to load source cloud, skipping..." << std::endl;
            continue;
        }

        std::cout << "Loading target: " << target_path << std::endl;
        if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(target_path, *target_cloud) == -1)
        {
            std::cerr << "  Failed to load target cloud, skipping..." << std::endl;
            continue;
        }

        std::cout << "  Source: " << source_cloud->size() << " points" << std::endl;
        std::cout << "  Target: " << target_cloud->size() << " points" << std::endl;

        // Ground-truth transform (source -> target)
        Eigen::Matrix4f ground_truth = pair.gt_matrix;

        if (config.voxel_grid_leaf_size > 0.0f)
        {
            source_cloud = voxelGridFilter(source_cloud, config.voxel_grid_leaf_size);
            target_cloud = voxelGridFilter(target_cloud, config.voxel_grid_leaf_size);
            std::cout << "  After downsampling: source=" << source_cloud->size()
                      << ", target=" << target_cloud->size() << std::endl;
        }

        int total_points = static_cast<int>(source_cloud->size() + target_cloud->size());

        float feature_radius = config.feature_radius;
        if (feature_radius <= 0.0f)
        {
            feature_radius = static_cast<float>(estimateSearchRadius(
                source_cloud, config.auto_radius_multiplier));
            std::cout << "  Auto-estimated feature radius: " << feature_radius << " m" << std::endl;
        }

        EvalConfig temp_config = config;
        temp_config.feature_radius = feature_radius;

        auto source_normals = estimateNormals(source_cloud, config.normal_k_search);
        auto target_normals = estimateNormals(target_cloud, config.normal_k_search);

        // 1. FPFH + RANSAC
        EvalResult fpfh_result = evaluateFPFH(source_cloud, target_cloud,
                                              source_normals, target_normals,
                                              ground_truth, temp_config, false);
        fpfh_result.pair_id = pair_id;
        fpfh_result.num_points = total_points;
        fpfh_result.fmr_success = (fpfh_result.inlier_ratio > config.fmr_inlier_ratio_threshold);
        fpfh_result.rr_success = (fpfh_result.success &&
                                  fpfh_result.rte < config.rr_rte_threshold &&
                                  fpfh_result.rre < config.rr_rre_threshold);
        all_results["FPFH + RANSAC"].push_back(fpfh_result);

        // 2. CSHOT + RANSAC
        EvalResult cshot_result = evaluateCSHOT(source_cloud, target_cloud,
                                                source_normals, target_normals,
                                                ground_truth, temp_config, false);
        cshot_result.pair_id = pair_id;
        cshot_result.num_points = total_points;
        cshot_result.fmr_success = (cshot_result.inlier_ratio > config.fmr_inlier_ratio_threshold);
        cshot_result.rr_success = (cshot_result.success &&
                                   cshot_result.rte < config.rr_rte_threshold &&
                                   cshot_result.rre < config.rr_rre_threshold);
        all_results["CSHOT + RANSAC"].push_back(cshot_result);

        // 3. ICP
        EvalResult icp_result = evaluateICP(source_cloud, target_cloud,
                                            ground_truth, temp_config,
                                            Eigen::Matrix4f::Identity(), false);
        icp_result.pair_id = pair_id;
        icp_result.num_points = total_points;
        icp_result.fmr_success = false;  // ICP has no feature matching
        icp_result.rr_success = (icp_result.success &&
                                 icp_result.rte < config.rr_rte_threshold &&
                                 icp_result.rre < config.rr_rre_threshold);
        all_results["ICP"].push_back(icp_result);

        // Print per-pair summary
        std::cout << "\n  Results for " << pair_id << ":" << std::endl;
        std::cout << "  " << std::string(60, '-') << std::endl;
        std::cout << "  " << std::left << std::setw(18) << "Method"
                  << std::right << std::setw(10) << "RRE(deg)"
                  << std::setw(10) << "RTE(m)"
                  << std::setw(10) << "IR(%)"
                  << std::setw(12) << "Time(ms)"
                  << std::endl;
        std::cout << "  " << std::string(60, '-') << std::endl;

        for (const std::string &method : {"FPFH + RANSAC", "CSHOT + RANSAC", "ICP"})
        {
            const EvalResult &r = all_results[method].back();
            std::cout << "  " << std::left << std::setw(18) << method
                      << std::right << std::fixed << std::setprecision(4)
                      << std::setw(10) << r.rre
                      << std::setw(10) << r.rte;
            if (r.inlier_ratio >= 0)
                std::cout << std::setw(10) << std::setprecision(2) << (r.inlier_ratio * 100.0);
            else
                std::cout << std::setw(10) << "N/A";
            std::cout << std::setw(12) << std::setprecision(2) << r.time_ms << std::endl;
        }
    }

    // Aggregate statistics (only when metrics are enabled)
    if (config.compute_metrics)
    {
        for (auto &pair : stats_map)
        {
            const std::string &method = pair.first;
            MethodStats &stats = pair.second;

            const std::vector<EvalResult> &results = all_results[method];
            stats.total_pairs = static_cast<int>(results.size());
            int n = stats.total_pairs;

            std::vector<double> rre_values, rte_values, rmse_values, reg_ir_values;

            for (const EvalResult &r : results)
            {
                if (r.success)
                {
                    rre_values.push_back(r.rre);
                    rte_values.push_back(r.rte);
                }
                if (r.inlier_rmse >= 0)
                {
                    rmse_values.push_back(r.inlier_rmse);
                }
                if (r.registration_ir >= 0)
                {
                    reg_ir_values.push_back(r.registration_ir);
                }
                if (r.fmr_success) stats.fmr_count++;
                if (r.rr_success) stats.rr_count++;
                stats.total_time_ms += r.time_ms;
                stats.total_points += r.num_points;
            }

            int success_count = static_cast<int>(rre_values.size());
            int rmse_count = static_cast<int>(rmse_values.size());
            int reg_ir_count = static_cast<int>(reg_ir_values.size());

            double sum_rre = 0.0, sum_rte = 0.0;
            for (size_t i = 0; i < rre_values.size(); ++i)
            {
                sum_rre += rre_values[i];
                sum_rte += rte_values[i];
            }
            stats.avg_rre = (success_count > 0) ? sum_rre / success_count : 0.0;
            stats.avg_rte = (success_count > 0) ? sum_rte / success_count : 0.0;

            double sum_rmse = 0.0;
            for (size_t i = 0; i < rmse_values.size(); ++i)
            {
                sum_rmse += rmse_values[i];
            }
            stats.avg_inlier_rmse = (rmse_count > 0) ? sum_rmse / rmse_count : -1.0;

            double sum_reg_ir = 0.0;
            for (size_t i = 0; i < reg_ir_values.size(); ++i)
            {
                sum_reg_ir += reg_ir_values[i];
            }
            stats.avg_registration_ir = (reg_ir_count > 0) ? sum_reg_ir / reg_ir_count : -1.0;

            // Unbiased std (divide by n-1)
            if (success_count > 1)
            {
                double sum_sq_rre = 0.0, sum_sq_rte = 0.0;
                for (size_t i = 0; i < rre_values.size(); ++i)
                {
                    sum_sq_rre += (rre_values[i] - stats.avg_rre) * (rre_values[i] - stats.avg_rre);
                    sum_sq_rte += (rte_values[i] - stats.avg_rte) * (rte_values[i] - stats.avg_rte);
                }
                stats.std_rre = std::sqrt(sum_sq_rre / (success_count - 1));
                stats.std_rte = std::sqrt(sum_sq_rte / (success_count - 1));
            }

            if (rmse_count > 1)
            {
                double sum_sq_rmse = 0.0;
                for (size_t i = 0; i < rmse_values.size(); ++i)
                {
                    sum_sq_rmse += (rmse_values[i] - stats.avg_inlier_rmse) *
                                   (rmse_values[i] - stats.avg_inlier_rmse);
                }
                stats.std_inlier_rmse = std::sqrt(sum_sq_rmse / (rmse_count - 1));
            }

            if (reg_ir_count > 1)
            {
                double sum_sq_reg_ir = 0.0;
                for (size_t i = 0; i < reg_ir_values.size(); ++i)
                {
                    sum_sq_reg_ir += (reg_ir_values[i] - stats.avg_registration_ir) *
                                     (reg_ir_values[i] - stats.avg_registration_ir);
                }
                stats.std_registration_ir = std::sqrt(sum_sq_reg_ir / (reg_ir_count - 1));
            }

            stats.fmr = (n > 0) ? static_cast<double>(stats.fmr_count) / n : 0.0;
            stats.rr  = (n > 0) ? static_cast<double>(stats.rr_count)  / n : 0.0;

            // s/KPts = total_time(s) / total_points_in_kilo
            if (stats.total_points > 0)
            {
                stats.time_per_kpts = (stats.total_time_ms / 1000.0) / (stats.total_points / 1000.0);
            }
        }

        printBatchResults(stats_map);

        std::vector<float> ir_thresholds = {0.0005f, 0.001f, 0.002f, 0.003f, 0.005f,
                                            0.01f, 0.02f, 0.03f, 0.05f};
        printFMRByIRThreshold(all_results, ir_thresholds);

        std::vector<float> distance_thresholds = {0.01f, 0.02f, 0.03f, 0.05f, 0.10f,
                                                  0.15f, 0.20f, 0.30f, 0.50f};
        printFMRByDistanceThreshold(all_results, distance_thresholds, config.fmr_inlier_ratio_threshold);
    }

    saveAllRegistrationResults(all_results, config.data_directory);

    std::cout << "\nBatch evaluation completed." << std::endl;

    return 0;
}
