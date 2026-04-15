#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <opencv2/imgproc.hpp>
#include <std_msgs/msg/string.hpp>
#include <map>
#include <vector>

namespace {
inline void LabelToBGR(const uint8_t label, uint8_t &b, uint8_t &g, uint8_t &r) {
  // 0 is dynamic/filtered out by caller; 255 is static/unknown.
  if (label == 255U) {
    b = 180U;
    g = 180U;
    r = 180U;
    return;
  }
  // Deterministic pseudo-color for semantic ids.
  b = static_cast<uint8_t>((37U * label + 53U) % 256U);
  g = static_cast<uint8_t>((17U * label + 101U) % 256U);
  r = static_cast<uint8_t>((29U * label + 197U) % 256U);
}
}  // namespace

class SemanticPointCloudBuilder : public rclcpp::Node {
public:
  SemanticPointCloudBuilder() : Node("semantic_pointcloud_builder") {
    // 内参直接由 semantic_pointcloud_builder.yaml 提供。
    fx_ = this->declare_parameter<double>("fx", 535.4);
    fy_ = this->declare_parameter<double>("fy", 539.2);
    cx_ = this->declare_parameter<double>("cx", 320.1);
    cy_ = this->declare_parameter<double>("cy", 247.6);
    RCLCPP_INFO(this->get_logger(), "Camera intrinsics: fx=%.4f fy=%.4f cx=%.4f cy=%.4f", fx_, fy_,
                cx_, cy_);

    voxel_leaf_size_ = this->declare_parameter<double>("voxel_leaf_size", 0.05);
    depth_min_ = this->declare_parameter<double>("depth_min", 0.1);
    depth_max_ = this->declare_parameter<double>("depth_max", 8.0);
    world_align_roll_deg_ = this->declare_parameter<double>("world_align_roll_deg", -90.0);

    const auto rgb_topic = this->declare_parameter<std::string>("rgb_topic", "/camera/rgb/image_color");
    const auto depth_topic = this->declare_parameter<std::string>("depth_topic", "/camera/depth/image");
    const auto mask_topic = this->declare_parameter<std::string>("mask_topic", "/yolo/mask");
    const auto pose_topic = this->declare_parameter<std::string>("pose_topic", "/orb_slam3/camera_pose");
    const auto output_topic = this->declare_parameter<std::string>("output_topic", "/semantic_global_map");
    const auto excluded_labels =
        this->declare_parameter<std::vector<int64_t>>("excluded_labels", std::vector<int64_t>{0, 255});

    excluded_labels_.clear();
    for (const auto label : excluded_labels) {
      if (label >= 0 && label <= 255) {
        excluded_labels_.insert(static_cast<uint8_t>(label));
      }
    }
    if (excluded_labels_.empty()) {
      excluded_labels_.insert(static_cast<uint8_t>(0));
      excluded_labels_.insert(static_cast<uint8_t>(255));
    }

    global_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
    global_cloud_->reserve(200000);

    pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, 5);
    // Topology publisher: publishes per-frame instance topology as JSON string
    pub_topology_ = this->create_publisher<std_msgs::msg::String>("/semantic/topology", 10);

    rgb_sub_.subscribe(this, rgb_topic);
    depth_sub_.subscribe(this, depth_topic);
    mask_sub_.subscribe(this, mask_topic);
    pose_sub_.subscribe(this, pose_topic);

    sync_ = std::make_shared<Synchronizer>(SyncPolicy(20), rgb_sub_, depth_sub_, mask_sub_, pose_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.08));
    sync_->registerCallback(
      std::bind(&SemanticPointCloudBuilder::sync_callback, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

    RCLCPP_INFO(this->get_logger(),
                "SemanticPointCloudBuilder started. topics: rgb=%s depth=%s mask=%s pose=%s out=%s",
                rgb_topic.c_str(), depth_topic.c_str(), mask_topic.c_str(), pose_topic.c_str(),
                output_topic.c_str());
  }

private:
  using ImageMsg = sensor_msgs::msg::Image;
  using PoseMsg = geometry_msgs::msg::PoseStamped;
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg, ImageMsg, PoseMsg>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  void sync_callback(const ImageMsg::ConstSharedPtr &rgb_msg, const ImageMsg::ConstSharedPtr &depth_msg,
                     const ImageMsg::ConstSharedPtr &mask_msg,
                     const PoseMsg::ConstSharedPtr &pose_msg) {
    cv_bridge::CvImageConstPtr rgb_ptr;
    cv_bridge::CvImageConstPtr depth_ptr;
    cv_bridge::CvImageConstPtr mask_ptr;

    try {
      rgb_ptr = cv_bridge::toCvShare(rgb_msg, "bgr8");
      depth_ptr = cv_bridge::toCvShare(depth_msg, "32FC1");
      mask_ptr = cv_bridge::toCvShare(mask_msg, "mono8");
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_WARN(this->get_logger(), "cv_bridge conversion failed: %s", e.what());
      return;
    }

    const cv::Mat &rgb = rgb_ptr->image;
    const cv::Mat &depth = depth_ptr->image;
    const cv::Mat &mask = mask_ptr->image;

    if (rgb.empty() || depth.empty() || mask.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Received empty frame(s), skip this sync callback");
      return;
    }

    if (rgb.rows != depth.rows || rgb.cols != depth.cols || rgb.rows != mask.rows ||
        rgb.cols != mask.cols) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Frame size mismatch: rgb(%dx%d), depth(%dx%d), mask(%dx%d)", rgb.cols,
                           rgb.rows, depth.cols, depth.rows, mask.cols, mask.rows);
      return;
    }

    // PoseStamped 按 T_wc 使用，将局部点云从相机系变换到世界系。
    Eigen::Quaternionf q(static_cast<float>(pose_msg->pose.orientation.w),
                         static_cast<float>(pose_msg->pose.orientation.x),
                         static_cast<float>(pose_msg->pose.orientation.y),
                         static_cast<float>(pose_msg->pose.orientation.z));
    if (q.norm() < 1e-6f) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Invalid pose quaternion, skip frame");
      return;
    }
    q.normalize();

    Eigen::Affine3f t_wc = Eigen::Affine3f::Identity();
    t_wc.linear() = q.toRotationMatrix();
    t_wc.translation() = Eigen::Vector3f(static_cast<float>(pose_msg->pose.position.x),
                                         static_cast<float>(pose_msg->pose.position.y),
                                         static_cast<float>(pose_msg->pose.position.z));

    // Align camera-centric world frame to RViz-friendly z-up frame.
    if (std::abs(world_align_roll_deg_) > 1e-6) {
      const float roll_rad = static_cast<float>(world_align_roll_deg_ * M_PI / 180.0);
      const Eigen::AngleAxisf roll_align(roll_rad, Eigen::Vector3f::UnitX());
      t_wc = roll_align * t_wc;
    }

    // Step: compute connected components on valid semantic pixels (exclude dynamic=0 and background=255)
    cv::Mat valid_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (int r = 0; r < mask.rows; ++r) {
      const uint8_t* mrow = mask.ptr<uint8_t>(r);
      uint8_t* vrow = valid_mask.ptr<uint8_t>(r);
      for (int c = 0; c < mask.cols; ++c) {
        const uint8_t mv = mrow[c];
        if (mv != 0 && mv != 255) vrow[c] = 255;
      }
    }

    cv::Mat labels, stats, centroids;
    int num_instances = cv::connectedComponentsWithStats(valid_mask, labels, stats, centroids, 8, CV_32S);

    // Containers to collect per-instance 3D points and semantic class
    std::map<int, std::vector<Eigen::Vector3f>> instance_points;
    std::map<int, uint32_t> instance_semantic_class;


    auto local_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
    local_cloud->points.reserve(static_cast<size_t>(rgb.rows * rgb.cols / 2));

    for (int v = 0; v < depth.rows; ++v) {
      const auto *depth_row = depth.ptr<float>(v);
      const auto *mask_row = mask.ptr<uint8_t>(v);

      for (int u = 0; u < depth.cols; ++u) {
        const float z = depth_row[u];
        if (!std::isfinite(z) || z <= static_cast<float>(depth_min_) ||
            z >= static_cast<float>(depth_max_)) {
          continue;
        }

        const uint8_t mask_value = mask_row[u];
        // combined mask协议: 0~254为类别ID, 255为背景；排除逻辑由excluded_labels控制。
        if (excluded_labels_.count(mask_value) > 0) {
          continue;
        }

        pcl::PointXYZRGBL pt;
        pt.z = z;
        pt.x = static_cast<float>((static_cast<double>(u) - cx_) * z / fx_);
        pt.y = static_cast<float>((static_cast<double>(v) - cy_) * z / fy_);
        pt.label = static_cast<uint32_t>(mask_value);
        LabelToBGR(mask_value, pt.b, pt.g, pt.r);

        // Collect per-instance 3D points using connected-component labels computed earlier.
        int instance_id = 0;
        if (!labels.empty() && labels.rows == mask.rows && labels.cols == mask.cols) {
          instance_id = labels.at<int>(v, u);
        }
        if (instance_id > 0) {
          // 这里是关键！必须乘上 t_wc 才能得到绝对世界坐标！
          Eigen::Vector3f pt_cam(pt.x, pt.y, pt.z);
          Eigen::Vector3f pt_world = t_wc * pt_cam;
          instance_points[instance_id].push_back(pt_world);
          instance_semantic_class[instance_id] = static_cast<uint32_t>(mask_value);
        }

        local_cloud->points.push_back(pt);
      }
    }

    if (local_cloud->points.empty()) {
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Local cloud is empty after filtering");
      return;
    }

    local_cloud->width = static_cast<uint32_t>(local_cloud->points.size());
    local_cloud->height = 1;
    local_cloud->is_dense = false;

    // Process collected per-instance points to compute centroid and AABB, then publish JSON.
    if (!instance_points.empty()) {
      std::string json = "{\"instances\": [";
      bool first_inst = true;
      for (const auto &kv : instance_points) {
        int inst_id = kv.first;
        const auto &pts = kv.second;
        if (pts.size() < 50) continue; // filter small/noisy instances

        // compute centroid and AABB
        float sumx = 0.0f, sumy = 0.0f, sumz = 0.0f;
        float minx = std::numeric_limits<float>::infinity();
        float miny = std::numeric_limits<float>::infinity();
        float minz = std::numeric_limits<float>::infinity();
        float maxx = -std::numeric_limits<float>::infinity();
        float maxy = -std::numeric_limits<float>::infinity();
        float maxz = -std::numeric_limits<float>::infinity();
        for (const auto &p : pts) {
          sumx += p.x(); sumy += p.y(); sumz += p.z();
          if (p.x() < minx) minx = p.x();
          if (p.y() < miny) miny = p.y();
          if (p.z() < minz) minz = p.z();
          if (p.x() > maxx) maxx = p.x();
          if (p.y() > maxy) maxy = p.y();
          if (p.z() > maxz) maxz = p.z();
        }
        const float n = static_cast<float>(pts.size());
        const float cx = sumx / n;
        const float cy = sumy / n;
        const float cz = sumz / n;

        uint32_t sem = 0;
        auto itc = instance_semantic_class.find(inst_id);
        if (itc != instance_semantic_class.end()) sem = itc->second;

        if (!first_inst) json += ", ";
        first_inst = false;
        json += "{\"instance_id\": ";
        json += std::to_string(inst_id);
        json += ", \"semantic_id\": ";
        json += std::to_string(sem);
        json += ", \"centroid\": [";
        json += std::to_string(cx) + "," + std::to_string(cy) + "," + std::to_string(cz);
        json += "], \"aabb_min\": [";
        json += std::to_string(minx) + "," + std::to_string(miny) + "," + std::to_string(minz);
        json += "], \"aabb_max\": [";
        json += std::to_string(maxx) + "," + std::to_string(maxy) + "," + std::to_string(maxz);
        json += "]}";
      }
      json += "]}";

      std_msgs::msg::String out;
      out.data = json;
      pub_topology_->publish(out);
    }

    auto world_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
    pcl::transformPointCloud(*local_cloud, *world_cloud, t_wc.matrix());

    *global_cloud_ += *world_cloud;

    pcl::VoxelGrid<pcl::PointXYZRGBL> voxel;
    voxel.setInputCloud(global_cloud_);
    voxel.setLeafSize(static_cast<float>(voxel_leaf_size_), static_cast<float>(voxel_leaf_size_),
                      static_cast<float>(voxel_leaf_size_));

    auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
    voxel.filter(*filtered);
    global_cloud_.swap(filtered);

    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*global_cloud_, out_msg);
    out_msg.header.stamp = pose_msg->header.stamp;
    out_msg.header.frame_id = "world";
    pub_cloud_->publish(out_msg);
  }

  double fx_{};
  double fy_{};
  double cx_{};
  double cy_{};
  double voxel_leaf_size_{};
  double depth_min_{};
  double depth_max_{};
  double world_align_roll_deg_{};
  std::unordered_set<uint8_t> excluded_labels_;

  message_filters::Subscriber<ImageMsg> rgb_sub_;
  message_filters::Subscriber<ImageMsg> depth_sub_;
  message_filters::Subscriber<ImageMsg> mask_sub_;
  message_filters::Subscriber<PoseMsg> pose_sub_;
  std::shared_ptr<Synchronizer> sync_;

  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr global_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_topology_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SemanticPointCloudBuilder>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
