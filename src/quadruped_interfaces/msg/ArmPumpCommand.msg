# 6-DOF arm + air pump command

float32[6] joint_positions  # target joint angles for 6-DOF manipulator [rad]
bool       pump_on          # air pump enable/disable
float32    pump_pressure    # target pressure (optional, 0 = ignore)
