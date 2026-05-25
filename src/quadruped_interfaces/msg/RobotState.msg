# Comprehensive robot state — published by serial_bridge, consumed by strategy_manager & nav stack
std_msgs/Header header

# ---------- IMU ----------
float32[3] orientation          # roll, pitch, yaw [rad]
float32[3] angular_velocity     # [rad/s]
float32[3] linear_acceleration  # [m/s²]

# ---------- Battery ----------
float32 battery_voltage   # [V]
float32 battery_percentage # [%]

# ---------- Foot contact ----------
bool[4] foot_contact  # true = in contact with ground (FL/FR/RL/RR)

# ---------- Leg joint feedback ----------
float32[12] leg_joint_positions
float32[12] leg_joint_velocities
float32[12] leg_joint_torques

# ---------- Arm joint feedback ----------
float32[6] arm_joint_positions

# ---------- System ----------
uint8   robot_state  # 0=idle, 1=walking, 2=manipulating, 3=error, 4=estop
uint32  error_code
string  error_msg
