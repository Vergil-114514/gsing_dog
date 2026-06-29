/**
 * @file    vision_transform.c
 * @brief   实现从相机米制坐标到机械臂毫米制坐标的自动转换
 */

#include "vision_transform.h"
#include <math.h>

// 实例化全局变量
volatile Arm_Pose_t vision_camera_data = {0}; // 接收上位机的米制数据
volatile Arm_Pose_t vision_world_data  = {0}; // 存储解算后的毫米制数据
volatile Arm_Pose_t debug_last_camera_data = {0};
volatile Arm_Pose_t debug_last_world_data = {0};
volatile uint32_t debug_vision_rx_count = 0;

void Perform_Vision_Coordinate_Transform(const Arm_Joint_Angles_t *angles,
                                         const Arm_Pose_t *current_end_pose)
{
    if (!angles || !current_end_pose) return;

    // 1. 先对 volatile 输入做一次快照，再将 m 转换为底层计算用的 mm。
    const float cam_x_mm = vision_camera_data.x * 1000.0f;
    const float cam_y_mm = vision_camera_data.y * 1000.0f;
    const float cam_z_mm = vision_camera_data.z * 1000.0f;
    debug_last_camera_data.x = vision_camera_data.x;
    debug_last_camera_data.y = vision_camera_data.y;
    debug_last_camera_data.z = vision_camera_data.z;
    debug_last_camera_data.pitch = vision_camera_data.pitch;

    // 2. 获取当前底座偏航角
    float theta1 = angles->theta1_geo;
    float cos_t1 = cosf(theta1);
    float sin_t1 = sinf(theta1);

    // 3. 局部坐标系映射：相机反装后，水平轴相对末端局部坐标需要翻转。
    // x_e (前方) = cam_y + 偏移
    // y_e (左侧) = cam_x + 偏移
    // z_e (上方) = -cam_z + 偏移 (因为 +Z 相机是下，所以取负变上)
    float x_e = cam_y_mm + CAMERA_OFFSET_X;
    float y_e = cam_x_mm + CAMERA_OFFSET_Y;
    float z_e = -cam_z_mm + CAMERA_OFFSET_Z;

    // 4. 旋转平移解算：将局部坐标变换到绝对基座坐标系
    vision_world_data.x = (x_e * cos_t1) - (y_e * sin_t1) + current_end_pose->x;
    vision_world_data.y = (x_e * sin_t1) + (y_e * cos_t1) + current_end_pose->y;
    vision_world_data.z = z_e + current_end_pose->z;

    vision_world_data.pitch = 0.0f;
    debug_last_world_data.x = vision_world_data.x;
    debug_last_world_data.y = vision_world_data.y;
    debug_last_world_data.z = vision_world_data.z;
    debug_last_world_data.pitch = vision_world_data.pitch;
    ++debug_vision_rx_count;
}
