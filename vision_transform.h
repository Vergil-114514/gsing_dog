/**
 * @file    vision_transform.h
 * @brief   深度相机视觉坐标到机械臂基座绝对坐标的转换模块 (支持米制输入)
 */

#ifndef __VISION_TRANSFORM_H
#define __VISION_TRANSFORM_H

#include "arm_kinematics.h"

// ==================== 相机安装偏置参数 (单位：mm) ====================
#define CAMERA_OFFSET_X   105.0f   // 相机光心相对于末端往前 70mm
#define CAMERA_OFFSET_Y   0.0f    // 相机左右偏置
#define CAMERA_OFFSET_Z  -78.0f   // 相机光心相对于末端往下 20mm

// ==================== 全局变量声明 ====================
/**
 * @brief 上位机实时传入的相机原始数据 (单位: m)
 * 你的通信协议解析后，将数据直接存入这个变量
 */
extern volatile Arm_Pose_t vision_camera_data;

/**
 * @brief 解算后的机械臂绝对坐标 (单位: mm)
 * 供你的控制逻辑或串口回传直接调用
 */
extern volatile Arm_Pose_t vision_world_data;

extern volatile Arm_Pose_t debug_last_camera_data;
extern volatile Arm_Pose_t debug_last_world_data;
extern volatile uint32_t debug_vision_rx_count;

// ==================== 函数声明 ====================

/**
 * @brief 核心解算函数：将 vision_camera_data 转换为 vision_world_data
 * @param angles            当前机械臂关节的真实几何角结构体 (用到 theta1_geo)
 * @param current_end_pose  当前机械臂末端的实时正解坐标 (current_arm_pose)
 */
void Perform_Vision_Coordinate_Transform(const Arm_Joint_Angles_t *angles,
                                         const Arm_Pose_t *current_end_pose);

#endif /* __VISION_TRANSFORM_H */
