"""
ROS 2 Image Publisher Node
功能：
1. 读取本地图片并以指定频率(15Hz)发布。
2. 使用 SensorDataQoS 保证低延迟。
3. 同时发布 /raw 和 /compressed 图像。
4. 支持 PNG 压缩等级配置。
"""

import os
import glob
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher_node')

        # --- 1. 参数声明与获取 ---
        self.declare_parameter('image_folder', '', ParameterDescriptor(description='图片文件夹绝对路径'))
        self.declare_parameter('publish_frequency', 15.0, ParameterDescriptor(description='发布频率 (Hz)'))
        self.declare_parameter('topic_name', 'camera/image_raw', ParameterDescriptor(description='基础话题名称'))
        self.declare_parameter('frame_id', 'camera_link', ParameterDescriptor(description='TF Frame ID'))
        self.declare_parameter('png_compression_level', 3, ParameterDescriptor(description='PNG压缩等级 (0-9), 3为平衡点'))

        self.img_folder = self.get_parameter('image_folder').value
        self.freq = self.get_parameter('publish_frequency').value
        self.topic_name = self.get_parameter('topic_name').value
        self.frame_id = self.get_parameter('frame_id').value
        self.png_level = self.get_parameter('png_compression_level').value

        # --- 2. 图像加载逻辑 ---
        self.images_path = []
        if not os.path.exists(self.img_folder):
            self.get_logger().error(f"路径不存在: {self.img_folder}")
        else:
            # 支持 jpg 和 png
            types = ('*.jpg', '*.jpeg', '*.png') 
            for files in types:
                self.images_path.extend(glob.glob(os.path.join(self.img_folder, files)))
            self.images_path.sort() # 排序保证播放顺序
            
        if not self.images_path:
            self.get_logger().warn(f"在 {self.img_folder} 未找到图片，将发布空数据或等待...")
        else:
            self.get_logger().info(f"成功加载 {len(self.images_path)} 张图片。")

        self.current_index = 0
        self.bridge = CvBridge()

        # --- 3. QoS 配置 (性能关键) ---
        # 使用 SensorDataQoS (Best Effort, Volatile) 以降低网络拥塞时的延迟
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- 4. 初始化发布者 ---
        # 原始图像发布者
        self.pub_raw = self.create_publisher(Image, self.topic_name, qos_profile)
        
        # 压缩图像发布者 (模拟 image_transport 的 compressed 话题)
        # 注意：标准 image_transport 插件的话题通常是 base_topic/compressed
        self.pub_compressed = self.create_publisher(
            CompressedImage, 
            f'{self.topic_name}/compressed', 
            qos_profile
        )

        # --- 5. 定时器 ---
        timer_period = 1.0 / self.freq
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"节点已启动，频率: {self.freq}Hz, PNG压缩等级: {self.png_level}")

    def timer_callback(self):
        if not self.images_path:
            return

        # 获取当前图片路径
        img_path = self.images_path[self.current_index]
        
        # 读取图片 (OpenCV format)
        cv_img = cv2.imread(img_path)
        
        if cv_img is None:
            self.get_logger().warn(f"无法读取图片: {img_path}")
            self._advance_index()
            return

        timestamp = self.get_clock().now().to_msg()

        # --- A. 发布 Raw Image ---
        try:
            msg_raw = self.bridge.cv2_to_imgmsg(cv_img, "bgr8")
            msg_raw.header.stamp = timestamp
            msg_raw.header.frame_id = self.frame_id
            self.pub_raw.publish(msg_raw)
        except Exception as e:
            self.get_logger().error(f"Raw转换失败: {e}")

        # --- B. 发布 Compressed Image (手动实现以精确控制 PNG 参数) ---
        try:
            msg_compressed = CompressedImage()
            msg_compressed.header.stamp = timestamp
            msg_compressed.header.frame_id = self.frame_id
            msg_compressed.format = "png"

            # 设置 PNG 压缩参数 [IMWRITE_PNG_COMPRESSION, level]
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), self.png_level]
            success, encoded_img = cv2.imencode('.png', cv_img, encode_param)

            if success:
                msg_compressed.data = encoded_img.tobytes()
                self.pub_compressed.publish(msg_compressed)
            else:
                self.get_logger().error("图像压缩失败")

        except Exception as e:
            self.get_logger().error(f"压缩发布失败: {e}")

        # 日志输出 (仅每15帧输出一次，避免刷屏)
        if self.current_index % 15 == 0:
            self.get_logger().info(f"Publishing: {os.path.basename(img_path)}")

        self._advance_index()

    def _advance_index(self):
        self.current_index += 1
        if self.current_index >= len(self.images_path):
            self.current_index = 0  # 循环播放

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()