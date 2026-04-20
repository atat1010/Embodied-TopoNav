#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped # 导入导航目标消息
from cv_bridge import CvBridge
import cv2
import base64
import threading
import os
import json
import re
from openai import OpenAI

class VLMBrainNode(Node):
    def __init__(self):
        super().__init__('vlm_brain_node')

        # YOLOv8 (COCO数据集) 80分类标准映射表
        self.coco_classes = {
            0: '人(person)', 1: '自行车(bicycle)', 2: '汽车(car)', 3: '摩托车(motorcycle)', 4: '飞机(airplane)',
            5: '公交车(bus)', 6: '火车(train)', 7: '卡车(truck)', 8: '船(boat)', 9: '红绿灯(traffic light)',
            10: '消防栓(fire hydrant)', 11: '停车牌(stop sign)', 12: '停车计时器(parking meter)', 13: '长椅(bench)', 14: '鸟(bird)',
            15: '猫(cat)', 16: '狗(dog)', 17: '马(horse)', 18: '羊(sheep)', 19: '牛(cow)',
            20: '大象(elephant)', 21: '熊(bear)', 22: '斑马(zebra)', 23: '长颈鹿(giraffe)', 24: '背包(backpack)',
            25: '雨伞(umbrella)', 26: '手提包(handbag)', 27: '领带(tie)', 28: '手提箱(suitcase)', 29: '飞盘(frisbee)',
            30: '滑雪板(skis)', 31: '单板滑雪(snowboard)', 32: '运动球(sports ball)', 33: '风筝(kite)', 34: '棒球棒(baseball bat)',
            35: '棒球手套(baseball glove)', 36: '滑板(skateboard)', 37: '冲浪板(surfboard)', 38: '网球拍(tennis racket)', 39: '瓶子(bottle)',
            40: '红酒杯(wine glass)', 41: '杯子(cup)', 42: '叉子(fork)', 43: '刀(knife)', 44: '勺子(spoon)',
            45: '碗(bowl)', 46: '香蕉(banana)', 47: '苹果(apple)', 48: '三明治(sandwich)', 49: '橙子(orange)',
            50: '西兰花(broccoli)', 51: '胡萝卜(carrot)', 52: '热狗(hot dog)', 53: '披萨(pizza)', 54: '甜甜圈(donut)',
            55: '蛋糕(cake)', 56: '椅子(chair)', 57: '沙发(couch)', 58: '盆栽(potted plant)', 59: '床(bed)',
            60: '餐桌(dining table)', 61: '马桶(toilet)', 62: '显示器(tv/monitor)', 63: '笔记本电脑(laptop)', 64: '鼠标(mouse)',
            65: '遥控器(remote)', 66: '键盘(keyboard)', 67: '手机(cell phone)', 68: '微波炉(microwave)', 69: '烤箱(oven)',
            70: '烤面包机(toaster)', 71: '洗手池(sink)', 72: '冰箱(refrigerator)', 73: '书(book)', 74: '时钟(clock)',
            75: '花瓶(vase)', 76: '剪刀(scissors)', 77: '泰迪熊(teddy bear)', 78: '吹风机(hair drier)', 79: '牙刷(toothbrush)'
        }

        # --- [1. 初始化 API 客户端] ---
        env_api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not env_api_key:
            raise ValueError("Missing API Key")

        self.declare_parameter('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        self.declare_parameter('model_name', 'qwen3-omni-flash')
        self.client = OpenAI(api_key=env_api_key, base_url=self.get_parameter('base_url').value)
        self.model_name = self.get_parameter('model_name').value

        self.bridge = CvBridge()
        
        # --- [2. 核心内存：长时语义地图] ---
        # 格式: {instance_id: {semantic_id, centroid, aabb, last_seen_stamp}}
        self.object_memory = {}
        self.latest_image_b64 = None
        
        # --- [3. 发布器：对接 Nav2] ---
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # --- [4. 订阅器] ---
        self.create_subscription(Image, '/camera/rgb/image_color', self.image_callback, 10)
        self.create_subscription(String, '/semantic/topology', self.topology_callback, 10)
        self.create_subscription(String, '/vlm/prompt', self.prompt_callback, 10)

        self.get_logger().info(f"具身大脑已就绪。正在监听拓扑数据进行记忆构建...")

    def topology_callback(self, msg):
        """持续更新 Python 端的‘世界模型’"""
        try:
            data = json.loads(msg.data)
            for inst in data['instances']:
                # 即使 C++ 那边有 GC，Python 也可以根据需要在这里做持久化存储
                self.object_memory[inst['instance_id']] = inst
        except Exception as e:
            self.get_logger().error(f"解析拓扑 JSON 失败: {e}")

    def image_callback(self, msg):
        # 图像处理保持不变（压缩并转 base64）
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        _, buffer = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.latest_image_b64 = base64.b64encode(buffer).decode('utf-8')

    def prompt_callback(self, msg):
        user_query = msg.data
        if not self.latest_image_b64 or not self.object_memory:
            self.get_logger().warn("数据尚未就绪...")
            return
        
        threading.Thread(target=self.execute_reasoning, args=(user_query,)).start()

    def execute_reasoning(self, query):
        # --- 新增的语义翻译中间件逻辑 ---
        translated_memory = []
        for inst_id, inst_data in self.object_memory.items():
            sem_id = inst_data.get('semantic_id')
            # 查字典，查不到给个默认值
            class_name = self.coco_classes.get(sem_id, f"未知物体_{sem_id}") 
            
            # 创建一个大模型专属的新字典，把干瘪的 semantic_id 替换成直观的 class_name
            translated_inst = {
                "instance_id": inst_id,
                "class_name": class_name,  # 核心改动：大模型直接看这个名字！
                "centroid": inst_data.get('centroid'),
                "aabb_min": inst_data.get('aabb_min'),
                "aabb_max": inst_data.get('aabb_max')
            }
            translated_memory.append(translated_inst)

        # 将当前的内存转换为 LLM 易读的简写形式（节省 token）
        memory_snapshot = json.dumps(translated_memory, indent=2, ensure_ascii=False)

        system_instructions = (
            "你是一个具身智能机器人的决策大脑。严格遵守以下空间坐标系规则（FLU）：\n"
            "- X轴（深度）：值越大越远。\n"
            "- Y轴（左右）：【绝对警告】Y为正数(+)代表在左边！Y为负数(-)代表在右边！注意：正负号决定左右，正数是左，负数是右！\n"
            "- Z轴（高度）：值越大越高。\n\n"
            "【任务规则】：\n"
            "1. 面对寻找特定方位物体（如左边、右边）的指令，你必须严格提取所有候选物体的 Y 轴坐标。\n"
            "2. 你【必须】严格按照以下格式输出，绝对不能省略 [思考] 环节：\n\n"
            "[思考] 这里写出你的比较过程。例如：候选A的Y值是xx，候选B的Y值是xx。因为要求找右侧，也就是Y值更小的（负数），所以选xx。\n"
            "[动作] \n"
            "```json\n"
            "{\"action\": \"navigate\", \"target_instance_id\": ID}\n"
            "```"
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"用户指令: {query}\n\n当前全局语义记忆: {memory_snapshot}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.latest_image_b64}"}}
                    ]}
                ]
            )

            response = completion.choices[0].message.content
            self.get_logger().info(f"大模型回复: {response}")
            
            # --- [5. 解析动作并执行导航] ---
            self.parse_and_execute_action(response)

        except Exception as e:
            self.get_logger().error(f"推理失败: {e}")

    def parse_and_execute_action(self, response):
        """从大模型的文本回复中提取 JSON 动作指令"""
        try:
            # 使用正则匹配 JSON 代码块
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                action_data = json.loads(match.group(1))
                if action_data.get("action") == "navigate":
                    target_id = action_data.get("target_instance_id")
                    self.send_navigation_goal(target_id)
        except Exception as e:
            self.get_logger().warn(f"解析动作指令失败: {e}")

    def send_navigation_goal(self, instance_id):
        """将物体坐标转换为导航点并发送给 Nav2"""
        obj = self.object_memory.get(instance_id)
        if not obj:
            self.get_logger().error(f"错误：ID {instance_id} 不在记忆中！")
            return

        # 提取质心 (FLU 坐标系: X前, Y左, Z上)
        cx, cy, cz = obj['centroid']
        
        # --- [核心逻辑：计算安全落脚点] ---
        # 我们不能导航到物体的物理质心（会撞上去），通常需要后退 0.6-0.8 米
        # 假设机器人需要从 X 方向靠近物体
        safe_offset = 0.7 
        goal_x = cx - safe_offset
        goal_y = cy

        # 构建 ROS 2 导航消息
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "world" # 必须与你 SLAM 的全局系一致
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        
        goal_msg.pose.position.x = float(goal_x)
        goal_msg.pose.position.y = float(goal_y)
        goal_msg.pose.position.z = 0.0 # 导航是在地面上跑
        
        # 简单计算朝向：让机器人面向物体
        goal_msg.pose.orientation.w = 1.0 # 简化处理，实际可以使用 atan2 计算 yaw

        self.get_logger().info(f"🚀 发送导航目标: 前往物体 {instance_id}，坐标为 ({goal_x:.2f}, {goal_y:.2f})")
        self.goal_pub.publish(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VLMBrainNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()