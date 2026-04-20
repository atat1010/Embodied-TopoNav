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
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <deque>
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
    // world_align_roll_deg_ = this->declare_parameter<double>("world_align_roll_deg", -90.0);

    const auto rgb_topic = this->declare_parameter<std::string>("rgb_topic", "/camera/rgb/image_color");
    const auto depth_topic = this->declare_parameter<std::string>("depth_topic", "/camera/depth/image");
    const auto mask_topic = this->declare_parameter<std::string>("mask_topic", "/yolo/mask");
    const auto pose_topic = this->declare_parameter<std::string>("pose_topic", "/orb_slam3/camera_pose");
    const auto output_topic = this->declare_parameter<std::string>("output_topic", "/semantic_global_map");
    const auto excluded_labels =
        this->declare_parameter<std::vector<int64_t>>("excluded_labels", std::vector<int64_t>{0});

    excluded_labels_.clear();
    for (const auto label : excluded_labels) {
      if (label >= 0 && label <= 255) {
        excluded_labels_.insert(static_cast<uint8_t>(label));
      }
    }
    if (excluded_labels_.empty()) {
      excluded_labels_.insert(static_cast<uint8_t>(0));
      // excluded_labels_.insert(static_cast<uint8_t>(255));
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
  struct GlobalInstance {
      int id;               // 永久唯一的实例 ID
      uint32_t semantic_id; // 语义类别 (如 56, 62)
      Eigen::Vector3f centroid;
      Eigen::Vector3f aabb_min;
      Eigen::Vector3f aabb_max;
      int hit_count = 0;    // 记录被看到的次数
      int miss_count = 0;   // <--- 【新增】连续未被看到的帧数
    };

  std::vector<GlobalInstance> global_instances_; // 全局实体备忘录
  int next_instance_id_ = 1;                     // 全局发号器
  
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
    // if (std::abs(world_align_roll_deg_) > 1e-6) {
    //   const float roll_rad = static_cast<float>(world_align_roll_deg_ * M_PI / 180.0);
    //   const Eigen::AngleAxisf roll_align(roll_rad, Eigen::Vector3f::UnitX());
    //   t_wc = roll_align * t_wc;
    // }

    // 引入标准的 ROS FLU 坐标系转换矩阵 
    // 目标：将 ORB-SLAM 的光学系 (Z前, X右, Y下) 
    // 转换为 Nav2 标准的全局系 (X前, Y左, Z上)
    Eigen::Matrix3f R_align;
    R_align <<  0,  0,  1,   // 新 X 轴 (前方) = 原 Z 轴
               -1,  0,  0,   // 新 Y 轴 (左方) = 原负 X 轴
                0, -1,  0;   // 新 Z 轴 (上方) = 原负 Y 轴

    Eigen::Affine3f T_align = Eigen::Affine3f::Identity();
    T_align.linear() = R_align;
    // 将对齐矩阵应用到当前位姿上
    t_wc = T_align * t_wc;
    // 坐标系对齐完成

    // Step: compute connected components on valid semantic pixels (exclude dynamic=0 and background=255)
    cv::Mat valid_mask = cv::Mat::zeros(mask.size(), CV_8UC1);

    for (int r = 0; r < mask.rows; ++r) {
      const uint8_t* mrow = mask.ptr<uint8_t>(r);
      uint8_t* vrow = valid_mask.ptr<uint8_t>(r);
      for (int c = 0; c < mask.cols; ++c) {
        const uint8_t mv = mrow[c];
        if (mv != 0 && mv != 255) vrow[c] = 255;    // 计算联通域需要剔除没被识别的背景，以及被识别但被我们排除的动态物体（如人）。剩下的才是我们关心的“合法语义像素”，用255标记。
        // 否则容易表现为联通域粘连过大，导致不同物体被错误地划分为同一个实例。
      }
    }
    // 2D 掩码边缘腐蚀 
    // 使用 5x5 的椭圆内核，将所有掩码向内收缩大约 2 个像素，直接扒掉危险的溢出边缘
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::erode(valid_mask, valid_mask, kernel);

    cv::Mat labels, stats, centroids;
    cv::connectedComponentsWithStats(valid_mask, labels, stats, centroids, 8, CV_32S);

    // Containers to collect per-instance 3D points and semantic class
    // std::map<int, std::vector<Eigen::Vector3f>> instance_points;
    // std::map<int, uint32_t> instance_semantic_class;


    auto local_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
    local_cloud->points.reserve(static_cast<size_t>(rgb.rows * rgb.cols / 2));

    for (int v = 0; v < depth.rows; ++v) {
      const auto *depth_row = depth.ptr<float>(v);
      const auto *mask_row = mask.ptr<uint8_t>(v);
      const auto *rgb_row = rgb.ptr<cv::Vec3b>(v);

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

        // --- ▼▼▼ 逻辑倒置：先查户口（是否被腐蚀掉了） ▼▼▼ ---
        int instance_id = 0;
        if (mask_value != 255) {
          if (!labels.empty() && labels.rows == mask.rows && labels.cols == mask.cols) {
            instance_id = labels.at<int>(v, u);
            // 如果这个像素在腐蚀操作中阵亡了（instance_id == 0），
            // 或者它本来就不属于任何连通域，直接抛弃！绝不加进点云！
            if (instance_id == 0) {
              continue; 
            }
          }
        }

        // --- ▼▼▼ 2.5D 深度悬崖过滤（扩大感受野应对深度斜坡） ▼▼▼ ---
        bool is_depth_edge = false;
        const float DEPTH_JUMP_THRESH = 0.20f; 
        
        // 跨步检查：不仅检查隔壁的 1 个像素，检查隔壁的 2 个像素！对抗相机的边缘插值斜坡
        if (v > 1 && v < depth.rows - 2 && u > 1 && u < depth.cols - 2) {
          float z_up    = depth.ptr<float>(v - 2)[u];
          float z_down  = depth.ptr<float>(v + 2)[u];
          float z_left  = depth_row[u - 2];
          float z_right = depth_row[u + 2];

          if (std::isfinite(z_up) && std::abs(z - z_up) > DEPTH_JUMP_THRESH) is_depth_edge = true;
          else if (std::isfinite(z_down) && std::abs(z - z_down) > DEPTH_JUMP_THRESH) is_depth_edge = true;
          else if (std::isfinite(z_left) && std::abs(z - z_left) > DEPTH_JUMP_THRESH) is_depth_edge = true;
          else if (std::isfinite(z_right) && std::abs(z - z_right) > DEPTH_JUMP_THRESH) is_depth_edge = true;
        }

        if (is_depth_edge) {
          continue; // 果断丢弃！
        }

        // --- 所有考验均通过，才允许正式成为 3D 世界的一员 ---
        pcl::PointXYZRGBL pt;
        pt.z = z;
        pt.x = static_cast<float>((static_cast<double>(u) - cx_) * z / fx_);
        pt.y = static_cast<float>((static_cast<double>(v) - cy_) * z / fy_);

        // 把 16 位的 instance_id 和 16 位的 mask_value 拼成一个 32 位整数塞进 label 里
        if (mask_value == 255) {
            pt.label = 255; // 背景点直接塞 255
        } 
        else {
        pt.label = (static_cast<uint32_t>(instance_id) << 16) | static_cast<uint32_t>(mask_value);
        }
        // LabelToBGR(mask_value, pt.b, pt.g, pt.r);
        const cv::Vec3b &bgr = rgb_row[u];
        pt.b = bgr[0];
        pt.g = bgr[1];
        pt.r = bgr[2];

        local_cloud->points.push_back(pt); // 只存入点云，先不去管 JSON 字典
      }
    }

    if (local_cloud->points.empty()) {
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "Local cloud is empty after filtering");
      return;
    }

    // 新增：3D 统计离群点移除 (SOR) 滤波
    // 专门用来剿灭悬浮在空中的“绝对飞点”和传感器噪声
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBL> sor;
    sor.setInputCloud(local_cloud);
    sor.setMeanK(50);            // 考察每个点周围的 50 个邻居
    sor.setStddevMulThresh(1.0); // 严格模式：距离标准差大于 1.0 的全部枪毙
    
    auto cleaned_local_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBL>>();
    sor.filter(*cleaned_local_cloud);
    
    local_cloud = cleaned_local_cloud; // 替换为干净的点云
    
    if (local_cloud->points.empty()) return;
    // SOR 滤波结束 

    // 新增：从洗干净的点云中，重新提取实例坐标（拆包）
    std::map<int, std::vector<Eigen::Vector3f>> instance_points;
    std::map<int, uint32_t> instance_semantic_class;

    for (auto &pt : local_cloud->points) {
      uint32_t inst_id = 0;
      uint32_t sem_id = pt.label;
      
      // 如果它不是背景，才去拆包提取 instance_id
      if (pt.label != 255) {
          inst_id = pt.label >> 16;       
          sem_id = pt.label & 0xFFFF;     
          
          // 核心防御：绝对不让背景（或者 0）去算 AABB 质心
          if (sem_id != 255 && inst_id > 0) {
              Eigen::Vector3f pt_cam(pt.x, pt.y, pt.z);
              Eigen::Vector3f pt_world = t_wc * pt_cam; 
              instance_points[inst_id].push_back(pt_world); 
              instance_semantic_class[inst_id] = sem_id;
          }
      }
      
        // 拆完包后，把 label 恢复成纯粹的语义 ID，防止 RViz 读不懂
      pt.label = sem_id; 
    }
    // 拆包提取结束

    local_cloud->width = static_cast<uint32_t>(local_cloud->points.size());
    local_cloud->height = 1;
    local_cloud->is_dense = false;

    // Process collected per-instance points to compute centroid and AABB, then publish JSON.
    if (!instance_points.empty()) {
      
      // --- 【新增】所有老熟人的“失踪计数器”先加 1 ---
      for (auto &inst : global_instances_) {
          inst.miss_count++;
      }

      for (const auto &kv : instance_points) {
        int temp_inst_id = kv.first;
        const auto &pts = kv.second;
        if (pts.size() < 50) continue; // 滤除小噪点

        // 1. 计算当前帧该物体的几何属性
        float sumx = 0, sumy = 0, sumz = 0;
        float minx = 1e6, miny = 1e6, minz = 1e6;
        float maxx = -1e6, maxy = -1e6, maxz = -1e6;
        for (const auto &p : pts) {
          sumx += p.x(); sumy += p.y(); sumz += p.z();
          if (p.x() < minx) minx = p.x(); 
          if (p.y() < miny) miny = p.y();
          if (p.z() < minz) minz = p.z();
          if (p.x() > maxx) maxx = p.x(); 
          if (p.y() > maxy) maxy = p.y(); 
          if (p.z() > maxz) maxz = p.z();
        }
        Eigen::Vector3f curr_centroid(sumx / pts.size(), sumy / pts.size(), sumz / pts.size());
        Eigen::Vector3f curr_min(minx, miny, minz);
        Eigen::Vector3f curr_max(maxx, maxy, maxz);
        
        uint32_t curr_sem = 0;
        if (instance_semantic_class.count(temp_inst_id)) curr_sem = instance_semantic_class[temp_inst_id];

        // 2. 去备忘录里找老熟人 (距离 < 0.5米 且 语义相同)
        bool matched = false;
        const float DISTANCE_THRESHOLD = 0.5f; 

        for (auto &global_inst : global_instances_) {
          if (global_inst.semantic_id == curr_sem) {
            float dist = (global_inst.centroid - curr_centroid).norm();
            if (dist < DISTANCE_THRESHOLD) {
              // 匹配成功：平滑更新质心，取最大的包围盒
              global_inst.centroid = 0.8f * global_inst.centroid + 0.2f * curr_centroid;

              // global_inst.aabb_min = global_inst.aabb_min.cwiseMin(curr_min);
              // global_inst.aabb_max = global_inst.aabb_max.cwiseMax(curr_max);

              // AABB 的正确更新方式：卡尔曼滤波思想 (EMA平滑)，而不是无限取极值！
              // 这允许包围盒在噪声消失后【自动收缩】回真实的物理尺寸
              global_inst.aabb_min = 0.8f * global_inst.aabb_min + 0.2f * curr_min;
              global_inst.aabb_max = 0.8f * global_inst.aabb_max + 0.2f * curr_max;
              global_inst.hit_count++;
              global_inst.miss_count = 0; // <--- 【新增】看到了就清零，续命成功！
              matched = true;
              break;
            }
          }
        }

        // 3. 没找到，说明是新物体，登记入库
        if (!matched) {
          GlobalInstance new_inst;
          new_inst.id = next_instance_id_++;
          new_inst.semantic_id = curr_sem;
          new_inst.centroid = curr_centroid;
          new_inst.aabb_min = curr_min;
          new_inst.aabb_max = curr_max;
          new_inst.hit_count = 1;
          global_instances_.push_back(new_inst);
        }
      }
    }

    // --- ▼▼▼ 【新增】架构升级：分级信用垃圾回收机制 ▼▼▼ ---
    global_instances_.erase(
        std::remove_if(global_instances_.begin(), global_instances_.end(),
            [](const GlobalInstance& inst) {
                if (inst.hit_count < 10) {
                    // 【考察期】：没看够 10 次，且丢失超过 30 帧（1.5秒）。直接抹杀！
                    return inst.miss_count > 30; 
                } else {
                    // 【转正期】：真实物体！绝对抗遮挡，永远存在于地图中！
                    return false; 
                }
            }),
        global_instances_.end()
    );
    // --- ▲▲▲ 垃圾回收结束 ▲▲▲ ---

    // 4. 发布完整的“全局备忘录”
    if (!global_instances_.empty()) {
      std::string json = "{\"instances\": [";
      bool first_inst = true;
      for (const auto &inst : global_instances_) {
        // 【修改】必须连续稳定看到 10 次，才配进入 JSON 输出！
        if (inst.hit_count < 10) continue; 

        if (!first_inst) json += ", ";
        first_inst = false;
        json += "{\"instance_id\": " + std::to_string(inst.id) +
                ", \"semantic_id\": " + std::to_string(inst.semantic_id) +
                ", \"forward_distance_m\": " + std::to_string(inst.centroid.x()) +
                ", \"centroid\": [" + std::to_string(inst.centroid.x()) + "," + std::to_string(inst.centroid.y()) + "," + std::to_string(inst.centroid.z()) + "]" +
                ", \"aabb_min\": [" + std::to_string(inst.aabb_min.x()) + "," + std::to_string(inst.aabb_min.y()) + "," + std::to_string(inst.aabb_min.z()) + "]" +
                ", \"aabb_max\": [" + std::to_string(inst.aabb_max.x()) + "," + std::to_string(inst.aabb_max.y()) + "," + std::to_string(inst.aabb_max.z()) + "]}";
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
  // double world_align_roll_deg_{};
  std::unordered_set<uint8_t> excluded_labels_;

  message_filters::Subscriber<ImageMsg> rgb_sub_;
  message_filters::Subscriber<ImageMsg> depth_sub_;
  message_filters::Subscriber<ImageMsg> mask_sub_;
  message_filters::Subscriber<PoseMsg> pose_sub_;
  std::shared_ptr<Synchronizer> sync_;

  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr global_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_topology_;

  // 新增：局部点云滑动窗口
  std::deque<pcl::PointCloud<pcl::PointXYZRGBL>::Ptr> cloud_queue_;
  const size_t MAX_CLOUD_FRAMES = 30; // 只保留最近 30 帧（约1-2秒）的视觉记忆

};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SemanticPointCloudBuilder>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
