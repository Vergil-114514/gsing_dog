# Leg control command — published by strategy_manager, consumed by serial_bridge
# 3 joints × 4 legs = 12 DOF

float32[12] joint_positions   # target joint angles [rad] (3 per leg: coxa/femur/tibia × 4)
float32[12] joint_velocities  # target joint velocities [rad/s] (0 = position mode)
uint8       control_mode       # 0=position, 1=velocity, 2=torque
