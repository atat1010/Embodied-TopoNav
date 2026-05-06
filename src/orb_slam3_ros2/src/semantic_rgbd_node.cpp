#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <mutex>
#include <string>
#include <deque>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <utility>
#include <unordered_set>
#include <Eigen/Geometry>

// 包含 ORB-SLAM3 的核心头文件
#include <System.h>

using namespace std::placeholders;

class SemanticRGBDNode : public rclcpp::Node {
public:
    SemanticRGBDNode(ORB_SLAM3::System* pSLAM)
        : Node("semantic_rgbd_node"), mSLAM(pSLAM) 
    {
        this->declare_parameter<std::string>("rgb_topic", "/camera/rgb/image_color");
        this->declare_parameter<std::string>("depth_topic", "/camera/depth/image");
        this->declare_parameter<std::string>("mask_topic", "/semantic/mask");
        this->declare_parameter<double>("mask_sync_tolerance", 0.08);
        this->declare_parameter<bool>("yolo_expected", false);
        this->declare_parameter<std::vector<int64_t>>("dynamic_labels", std::vector<int64_t>{0});

        const auto rgb_topic = this->get_parameter("rgb_topic").as_string();
        const auto depth_topic = this->get_parameter("depth_topic").as_string();
        const auto mask_topic = this->get_parameter("mask_topic").as_string();
        mask_sync_tolerance_sec_ = this->get_parameter("mask_sync_tolerance").as_double();
        mask_buffer_window_sec_ = std::max(0.5, 3.0 * mask_sync_tolerance_sec_);
        yolo_expected_ = this->get_parameter("yolo_expected").as_bool();
        const auto dynamic_labels = this->get_parameter("dynamic_labels").as_integer_array();
        dynamic_labels_.clear();
        for (const auto label : dynamic_labels) {
            if (label >= 0 && label <= 255) {
                dynamic_labels_.insert(static_cast<uint8_t>(label));
            }
        }
        if (dynamic_labels_.empty()) {
            dynamic_labels_.insert(static_cast<uint8_t>(0));
        }

        // 1. 初始化订阅器 (默认对齐 TUM ros2bag 话题，可由 launch 覆盖)
        rgb_sub_.subscribe(this, rgb_topic);
        depth_sub_.subscribe(this, depth_topic);
        mask_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            mask_topic,
            10,
            std::bind(&SemanticRGBDNode::MaskCallback, this, _1)
        );

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/orb_slam3/camera_pose", 10);
        fps_pub_ = this->create_publisher<std_msgs::msg::Float32>("/orb_slam3/fps", 10);

        // 2. 初始化时间同步器 (RGB和Depth的时间戳必须对齐才能给SLAM)
        sync_.reset(new Sync(MySyncPolicy(10), rgb_sub_, depth_sub_));
        sync_->registerCallback(std::bind(&SemanticRGBDNode::ImageCallback, this, _1, _2));

        RCLCPP_INFO(
            this->get_logger(),
            "节点已启动: rgb=%s depth=%s mask=%s pose_topic=%s fps_topic=%s",
            rgb_topic.c_str(),
            depth_topic.c_str(),
            mask_topic.c_str(),
            "/orb_slam3/camera_pose",
            "/orb_slam3/fps"
        );
        RCLCPP_INFO(this->get_logger(), "mask_sync_tolerance: %.3f s", mask_sync_tolerance_sec_);
        RCLCPP_INFO(this->get_logger(), "yolo_expected: %s", yolo_expected_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "dynamic_labels for SLAM binary mask size: %zu", dynamic_labels_.size());
    }

private:
    void ImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msgRGB,
                       const sensor_msgs::msg::Image::ConstSharedPtr& msgD) 
    {
        // 1. 将 ROS 图像转化为 OpenCV 矩阵
        cv_bridge::CvImageConstPtr cv_ptrRGB;
        cv_bridge::CvImageConstPtr cv_ptrD;
        try {
            cv_ptrRGB = cv_bridge::toCvShare(msgRGB, "bgr8");

            if (msgD->encoding == "16UC1") {
                cv_ptrD = cv_bridge::toCvShare(msgD, "16UC1");
            } else if (msgD->encoding == "32FC1") {
                cv_ptrD = cv_bridge::toCvShare(msgD, "32FC1");
            } else {
                cv_ptrD = cv_bridge::toCvShare(msgD, "passthrough");
            }
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge 异常: %s", e.what());
            return;
        }

        // 提取时间戳
        double timestamp = msgRGB->header.stamp.sec + msgRGB->header.stamp.nanosec * 1e-9;

        cv::Mat current_mask;
        bool has_valid_mask = false;
        const rclcpp::Time rgb_stamp(msgRGB->header.stamp);
        {
            std::lock_guard<std::mutex> lock(mask_mutex_);
            // 最近邻对齐：在缓存中找与当前 RGB 时间戳最接近的一帧 mask。
            if (!mask_buffer_.empty()) {
                int best_idx = -1;
                double best_dt = std::numeric_limits<double>::infinity();
                for (int i = 0; i < static_cast<int>(mask_buffer_.size()); ++i) {
                    const double dt = std::abs((rgb_stamp - mask_buffer_[i].stamp).seconds());
                    if (dt < best_dt) {
                        best_dt = dt;
                        best_idx = i;
                    }
                }
                if (best_idx >= 0 && best_dt <= mask_sync_tolerance_sec_) {
                    current_mask = mask_buffer_[best_idx].mask.clone();
                    has_valid_mask = true;
                }
            }
        }

        if (!has_valid_mask) {
            const auto now = this->now();
            if ((now - last_mask_mismatch_log_time_).seconds() >= 1.0) {
                RCLCPP_WARN(
                    this->get_logger(),
                    "mask 与当前RGB帧时间戳偏差超过阈值(%.3f s)本帧不使用mask",
                    mask_sync_tolerance_sec_
                );
                last_mask_mismatch_log_time_ = now;
            }
        }

        frame_mask_usage_.emplace_back(timestamp, has_valid_mask);
        while (!frame_mask_usage_.empty() && (timestamp - frame_mask_usage_.front().first) > 1.0) {
            frame_mask_usage_.pop_front();
        }

        cv::Mat depth_for_slam;
        if (cv_ptrD->image.type() == CV_16UC1 || cv_ptrD->image.type() == CV_32FC1) {
            depth_for_slam = cv_ptrD->image;
        } else {
            cv_ptrD->image.convertTo(depth_for_slam, CV_32FC1);
        }

        // 2. 喂给 ORB-SLAM3 的核心引擎进行跟踪（mask 可为空）
        Sophus::SE3f Tcw = mSLAM->TrackRGBD(cv_ptrRGB->image, depth_for_slam, timestamp, {}, "", current_mask);

        PublishPoseAndFps(Tcw, msgRGB->header, timestamp);
    }

    void PublishPoseAndFps(const Sophus::SE3f& Tcw, const std_msgs::msg::Header& header, double timestamp)
    {
        if (Tcw.matrix().allFinite()) {
            const Sophus::SE3f Twc = Tcw.inverse();
            const Eigen::Vector3f t = Twc.translation();
            const Eigen::Quaternionf q(Twc.rotationMatrix());

            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header = header;
            pose_msg.header.frame_id = "world";
            pose_msg.pose.position.x = static_cast<double>(t.x());
            pose_msg.pose.position.y = static_cast<double>(t.y());
            pose_msg.pose.position.z = static_cast<double>(t.z());
            pose_msg.pose.orientation.x = static_cast<double>(q.x());
            pose_msg.pose.orientation.y = static_cast<double>(q.y());
            pose_msg.pose.orientation.z = static_cast<double>(q.z());
            pose_msg.pose.orientation.w = static_cast<double>(q.w());
            pose_pub_->publish(pose_msg);
        }

        frame_timestamps_.push_back(timestamp);
        while (!frame_timestamps_.empty() && (timestamp - frame_timestamps_.front()) > 1.0) {
            frame_timestamps_.pop_front();
        }

        std_msgs::msg::Float32 fps_msg;
        fps_msg.data = static_cast<float>(frame_timestamps_.size());
        fps_pub_->publish(fps_msg);

        const auto now = this->now();
        if ((now - last_fps_log_time_).seconds() >= 1.0) {
            size_t mask_count = 0;
            rclcpp::Time last_mask_rx_time = latest_mask_stamp_;
            {
                std::lock_guard<std::mutex> lock(mask_mutex_);
                while (!mask_arrival_timestamps_.empty() && (timestamp - mask_arrival_timestamps_.front()) > 1.0) {
                    mask_arrival_timestamps_.pop_front();
                }
                mask_count = mask_arrival_timestamps_.size();
                last_mask_rx_time = latest_mask_stamp_;
            }

            size_t used_count = 0;
            for (const auto& item : frame_mask_usage_) {
                if (item.second) {
                    ++used_count;
                }
            }
            const double used_ratio = frame_mask_usage_.empty()
                ? 0.0
                : (100.0 * static_cast<double>(used_count) / static_cast<double>(frame_mask_usage_.size()));

            const bool yolo_alive = last_mask_rx_time.nanoseconds() > 0 && (now - last_mask_rx_time).seconds() < 2.0;

            RCLCPP_INFO(
                this->get_logger(),
                "SLAM FPS(1s): %.1f | YOLO expected: %s | YOLO alive: %s | mask FPS(1s): %zu | mask used: %.1f%%",
                fps_msg.data,
                yolo_expected_ ? "true" : "false",
                yolo_alive ? "true" : "false",
                mask_count,
                used_ratio
            );

            if (yolo_expected_ && !yolo_alive) {
                RCLCPP_WARN(this->get_logger(), "YOLO expected=true 但最近2秒没有收到mask，可能YOLO未运行或推理过慢");
            }
            last_fps_log_time_ = now;
        }
    }

    void MaskCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msgMask)
    {
        try {
            cv_bridge::CvImageConstPtr cv_ptrMask;
            if (msgMask->encoding == "mono8")
                cv_ptrMask = cv_bridge::toCvShare(msgMask, "mono8");
            else
                cv_ptrMask = cv_bridge::toCvShare(msgMask, "passthrough");

            cv::Mat mask = cv_ptrMask->image;
            if(mask.empty())
                return;

            if(mask.channels() == 3)
                cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
            else if(mask.channels() == 4)
                cv::cvtColor(mask, mask, cv::COLOR_BGRA2GRAY);

            if(mask.type() != CV_8UC1)
                mask.convertTo(mask, CV_8UC1);

            // combined_mask后处理: dynamic_labels中的类别标为0，其余统一标为255供SLAM使用。
            cv::Mat slam_mask(mask.rows, mask.cols, CV_8UC1, cv::Scalar(255));
            for (int v = 0; v < mask.rows; ++v) {
                const auto* src = mask.ptr<uint8_t>(v);
                auto* dst = slam_mask.ptr<uint8_t>(v);
                for (int u = 0; u < mask.cols; ++u) {
                    if (dynamic_labels_.count(src[u]) > 0) {
                        dst[u] = 0;
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(mask_mutex_);
                latest_mask_ = slam_mask;
                latest_mask_stamp_ = rclcpp::Time(msgMask->header.stamp);
                const double mask_ts = latest_mask_stamp_.seconds();
                mask_arrival_timestamps_.push_back(mask_ts);

                mask_buffer_.push_back(TimedMask{latest_mask_stamp_, slam_mask});
                // 只保留最近一段时间窗口，既支持最近邻查找，也避免内存增长。
                while (!mask_buffer_.empty() &&
                       (latest_mask_stamp_ - mask_buffer_.front().stamp).seconds() > mask_buffer_window_sec_) {
                    mask_buffer_.pop_front();
                }
            }
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN(this->get_logger(), "mask cv_bridge 异常: %s", e.what());
        }
    }

    // 成员变量
    ORB_SLAM3::System* mSLAM;
    
    // ROS 2 消息同步相关
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mask_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr fps_pub_;
    
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    struct TimedMask {
        rclcpp::Time stamp;
        cv::Mat mask;
    };
    std::shared_ptr<Sync> sync_;
    std::mutex mask_mutex_;
    cv::Mat latest_mask_;
    rclcpp::Time latest_mask_stamp_{0, 0, RCL_ROS_TIME};
    std::deque<TimedMask> mask_buffer_;
    std::deque<double> mask_arrival_timestamps_;
    std::deque<double> frame_timestamps_;
    std::deque<std::pair<double, bool>> frame_mask_usage_;
    rclcpp::Time last_fps_log_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_mask_mismatch_log_time_{0, 0, RCL_ROS_TIME};
    double mask_sync_tolerance_sec_{0.08};
    double mask_buffer_window_sec_{1.0};
    bool yolo_expected_{false};
    std::unordered_set<uint8_t> dynamic_labels_;
};

int main(int argc, char **argv) {
    std::vector<std::string> args = rclcpp::init_and_remove_ros_arguments(argc, argv);

    // 检查输入参数 (需要字典文件和相机参数文件)
    if (args.size() != 3) {
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "用法: ros2 run orb_slam3_ros2 semantic_rgbd_node path_to_vocabulary path_to_settings");
        rclcpp::shutdown();
        return 1;
    }

    bool use_viewer = true;
    if (const char* env = std::getenv("ORB_SLAM3_USE_VIEWER")) {
        const std::string v(env);
        if (v == "0" || v == "false" || v == "False") {
            use_viewer = false;
        }
    }

    // 实例化 ORB-SLAM3 系统 (RGBD模式)
    ORB_SLAM3::System SLAM(args[1], args[2], ORB_SLAM3::System::RGBD, use_viewer);

    // 启动我们的 ROS 2 节点
    auto node = std::make_shared<SemanticRGBDNode>(&SLAM);
    rclcpp::spin(node);

    // 节点关闭前，先安全关闭 SLAM 系统
    SLAM.Shutdown();

    rclcpp::shutdown();

    return 0;
}