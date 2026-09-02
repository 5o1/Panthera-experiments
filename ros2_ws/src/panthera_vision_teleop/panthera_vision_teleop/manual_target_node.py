"""阶段 0：向 Panthera 发送一次微小相对位移的学习节点。

节点默认运行在 dry-run 模式，只发布目标预览；只有显式设置
``send_robot_command=true`` 后才会创建并使用真机命令发布者。

数据流：

    /end_pose_euler -> 本节点 -> /teleop/manual_target_preview
    /arm_status ----> 本节点 -> /pos_cmd（仅显式允许后发送一次）
"""

import math
from typing import Optional, Sequence

import rclpy
from rclpy.node import Node

from panthera_interfaces.msg import ArmStatus, EndPoseEuler, PosCmd


class ManualTargetNode(Node):
    """生成并选择性发送一次安全受限的末端相对位移目标。"""

    def __init__(self) -> None:
        """初始化 ROS 节点、参数、状态、订阅者和发布者。

        需要完成：
        1. 调用 ``Node`` 父类构造函数并设置节点名。
        2. 声明并读取 ``manual_test.yaml`` 中的全部参数。
        3. 调用 :meth:`validate_delta` 检查 ``delta_xyz``。
        4. 初始化最新末端位姿、机械臂状态和“是否已发送”标记。
        5. 创建末端位姿与机械臂状态订阅者。
        6. 创建 preview 与 robot command 发布者。

        注意：不得在构造函数中直接发送机器人命令。
        """
        super().__init__("manual_target_node")

        # 声明参数和安全默认值。--params-file 传入的 YAML 值会覆盖默认值。
        self.declare_parameter("send_robot_command", False)
        self.declare_parameter("delta_xyz", [0.01, 0.0, 0.0])
        self.declare_parameter("max_abs_delta_m", 0.02)
        self.declare_parameter("preview_topic", "/teleop/manual_target_preview")
        self.declare_parameter("robot_command_topic", "/pos_cmd")
        self.declare_parameter("end_pose_topic", "/end_pose_euler")
        self.declare_parameter("arm_status_topic", "/arm_status")
        self.declare_parameter("mode1", 0)
        self.declare_parameter("gripper", -1.0)

        # 从 ROS 参数系统读取最终值；不需要在 Python 中直接打开 YAML。
        self.send_robot_command = self.get_parameter(
            "send_robot_command"
        ).get_parameter_value().bool_value
        self.delta_xyz = list(
            self.get_parameter(
                "delta_xyz"
            ).get_parameter_value().double_array_value
        )
        self.max_abs_delta_m = self.get_parameter(
            "max_abs_delta_m"
        ).get_parameter_value().double_value
        self.preview_topic = self.get_parameter(
            "preview_topic"
        ).get_parameter_value().string_value
        self.robot_command_topic = self.get_parameter(
            "robot_command_topic"
        ).get_parameter_value().string_value
        self.end_pose_topic = self.get_parameter(
            "end_pose_topic"
        ).get_parameter_value().string_value
        self.arm_status_topic = self.get_parameter(
            "arm_status_topic"
        ).get_parameter_value().string_value
        self.mode1 = self.get_parameter(
            "mode1"
        ).get_parameter_value().integer_value
        self.gripper = self.get_parameter(
            "gripper"
        ).get_parameter_value().double_value

        self.delta_xyz = self.validate_delta(self.delta_xyz, self.max_abs_delta_m)

        if self.mode1 != 0:
            raise ValueError("阶段 0 只允许 mode1=0")
        if self.gripper != -1.0:
            raise ValueError("阶段 0 必须使用 gripper=-1.0，禁止改变夹爪状态")

        self.latest_end_pose: Optional[EndPoseEuler] = None
        self.latest_arm_status: Optional[ArmStatus] = None
        self.target_processed = False

        self.preview_publisher = self.create_publisher(
            PosCmd, self.preview_topic, 10
        )
        # dry-run 时不创建 /pos_cmd publisher，便于从 ROS graph 确认节点
        # 不具备向真机发送命令的能力。
        self.robot_command_publisher = (
            self.create_publisher(PosCmd, self.robot_command_topic, 10)
            if self.send_robot_command
            else None
        )
        self.end_pose_subscription = self.create_subscription(
            EndPoseEuler, self.end_pose_topic, self.end_pose_callback, 10
        )
        self.arm_status_subscription = self.create_subscription(
            ArmStatus, self.arm_status_topic, self.arm_status_callback, 10
        )

        mode = "ARMED" if self.send_robot_command else "DRY-RUN"
        self.get_logger().info(
            f"{mode}: delta_xyz={self.delta_xyz}, "
            f"max_abs_delta_m={self.max_abs_delta_m}"
        )


    def end_pose_callback(self, msg: EndPoseEuler) -> None:
        """保存驱动发布的最新末端位姿，并尝试生成一次目标。

        参数：
            msg: ``/end_pose_euler`` 收到的当前位置与 RPY 姿态。

        收到消息后应保存它，再调用 :meth:`try_publish_target`。不要在
        本函数里重复实现目标计算或安全判断。
        """
        self.latest_end_pose = msg
        self.try_publish_target()

    def arm_status_callback(self, msg: ArmStatus) -> None:
        """保存驱动发布的最新机械臂状态，并尝试生成一次目标。

        参数：
            msg: ``/arm_status`` 收到的使能、运动状态和故障信息。

        两个订阅话题的消息到达顺序不确定，因此这里也要调用
        :meth:`try_publish_target`。
        """
        self.latest_arm_status = msg
        self.try_publish_target()

    def validate_delta(
        self,
        delta_xyz: Sequence[float],
        max_abs_delta_m: float,
    ) -> tuple[float, float, float]:
        """验证并规范化三轴相对位移参数。

        参数：
            delta_xyz: 期望的 X/Y/Z 相对位移，单位为米。
            max_abs_delta_m: 每个轴允许的最大绝对偏移，单位为米。

        返回：
            长度固定为 3 的 ``(dx, dy, dz)`` 浮点数元组。

        需要拒绝：
        - 元素数量不是 3；
        - 包含 NaN 或无穷大；
        - ``max_abs_delta_m`` 非法；
        - 任意轴超过安全上限。

        非法输入应抛出 ``ValueError``。
        """
        if not math.isfinite(max_abs_delta_m) or max_abs_delta_m <= 0.0:
            raise ValueError("max_abs_delta_m 必须是大于 0 的有限数")

        if len(delta_xyz) != 3:
            raise ValueError("delta_xyz 必须恰好包含 3 个元素")

        try:
            delta = tuple(float(value) for value in delta_xyz)
        except (TypeError, ValueError) as exc:
            raise ValueError("delta_xyz 的三个元素必须是数值") from exc

        if not all(math.isfinite(value) for value in delta):
            raise ValueError("delta_xyz 不能包含 NaN 或无穷大")

        if any(abs(value) > max_abs_delta_m for value in delta):
            raise ValueError(
                "delta_xyz 的任意轴绝对值都不能超过 "
                f"{max_abs_delta_m} m"
            )

        return delta

    def build_target(self, current_pose: EndPoseEuler) -> PosCmd:
        """根据当前末端位姿构造一个微小相对移动目标。

        参数：
            current_pose: 最近一次收到的真实末端位姿。

        返回：
            待预览或发送的 Panthera ``PosCmd`` 消息。

        构造规则：
        - XYZ 等于当前位置加已经验证的 ``delta_xyz``；
        - roll、pitch、yaw 原样复制当前位置；
        - gripper 使用配置值，阶段 0 应为 ``-1.0``；
        - mode1 使用配置值；
        - mode2 使用接口保留值 ``0``。
        """
        pose_values = (
            current_pose.x,
            current_pose.y,
            current_pose.z,
            current_pose.roll,
            current_pose.pitch,
            current_pose.yaw,
        )
        if not all(math.isfinite(value) for value in pose_values):
            raise ValueError("当前末端位姿包含 NaN 或无穷大")

        target = PosCmd()
        target.x = current_pose.x + self.delta_xyz[0]
        target.y = current_pose.y + self.delta_xyz[1]
        target.z = current_pose.z + self.delta_xyz[2]
        target.roll = current_pose.roll
        target.pitch = current_pose.pitch
        target.yaw = current_pose.yaw
        target.gripper = self.gripper
        target.mode1 = self.mode1
        target.mode2 = 0
        return target

    def status_allows_motion(self, status: ArmStatus) -> tuple[bool, str]:
        """判断机械臂当前状态是否允许发送真机运动命令。

        参数：
            status: 最近一次收到的机械臂状态。

        返回：
            ``(allowed, reason)``。允许时 ``allowed`` 为 ``True``；拒绝时
            ``reason`` 应说明原因，供日志输出。

        至少检查机械臂已使能、当前空闲、没有总体错误、所有电机无
        故障且关节没有触及限位。
        """
        if not status.arm_enabled:
            return False, "机械臂未使能"
        if status.motion_status != 0:
            return False, f"机械臂不是空闲状态（motion_status={status.motion_status}）"
        if status.error_message.strip():
            return False, f"机械臂报告错误：{status.error_message}"
        if any(fault != 0 for fault in status.motor_faults):
            return False, f"电机故障码非零：{list(status.motor_faults)}"
        if any(status.joint_at_limit):
            return False, f"有关节触及限位：{list(status.joint_at_limit)}"
        if status.gripper_fault != 0:
            return False, f"夹爪故障码非零：{status.gripper_fault}"
        return True, "状态正常"

    def try_publish_target(self) -> None:
        """在输入齐全时预览目标，并按安全条件最多发送一次真机命令。

        推荐执行顺序：
        1. 若已经处理过一次目标，立即返回；
        2. 若末端位姿或机械臂状态尚未收到，等待后返回；
        3. 调用 :meth:`build_target` 构造目标；
        4. 始终向 preview 话题发布一次；
        5. 若 ``send_robot_command`` 为 false，记录 ``DRY-RUN`` 后结束；
        6. 调用 :meth:`status_allows_motion` 做真机安全检查；
        7. 通过检查后向 ``/pos_cmd`` 发布一次，并记录 ``SENT``。

        无论 dry-run、blocked 还是 sent，一个进程生命周期都只能处理
        一个目标，禁止由连续状态消息触发重复发布。
        """
        if self.target_processed:
            return
        if self.latest_end_pose is None or self.latest_arm_status is None:
            return

        # 在发布前锁定本次生命周期，防止后续状态消息造成重复发送。
        self.target_processed = True

        try:
            target = self.build_target(self.latest_end_pose)
        except ValueError as exc:
            self.get_logger().error(f"BLOCKED: 无法构造安全目标：{exc}")
            return

        self.preview_publisher.publish(target)

        if not self.send_robot_command:
            self.get_logger().info(
                "DRY-RUN: 已发布一次目标预览，未发布 /pos_cmd"
            )
            return

        allowed, reason = self.status_allows_motion(self.latest_arm_status)
        if not allowed:
            self.get_logger().error(f"BLOCKED: {reason}")
            return

        if self.robot_command_publisher is None:
            self.get_logger().error("BLOCKED: 真机 publisher 未创建")
            return

        self.robot_command_publisher.publish(target)
        self.get_logger().warning(
            "SENT: 已向 /pos_cmd 发布一次真机目标，后续消息不会再次发送"
        )


def main(args: Optional[Sequence[str]] = None) -> None:
    """初始化 rclpy，运行节点，并在退出时可靠释放 ROS 资源。

    需要完成：
    1. 调用 ``rclpy.init``；
    2. 创建 :class:`ManualTargetNode`；
    3. 使用 ``rclpy.spin`` 处理订阅回调；
    4. 正确处理 ``KeyboardInterrupt``；
    5. 在 ``finally`` 中销毁节点并关闭 rclpy。
    """
    rclpy.init(args=args)
    node: Optional[ManualTargetNode] = None
    try:
        node = ManualTargetNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
