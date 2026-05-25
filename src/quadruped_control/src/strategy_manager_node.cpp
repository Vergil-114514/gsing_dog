/**
 * strategy_manager_node — High-level coordination node
 *
 * Subscribes to RobotState (serial_bridge) and vision results.
 * Runs a simple state machine that decides what LegCommand / ArmPumpCommand
 * to publish.  Ready to be extended with BehaviorTree.CPP or a Nav2 action
 * client when navigation goals come into scope.
 */

#include <rclcpp/rclcpp.hpp>
#include <quadruped_interfaces/msg/leg_command.hpp>
#include <quadruped_interfaces/msg/arm_pump_command.hpp>
#include <quadruped_interfaces/msg/robot_state.hpp>
#include <std_msgs/msg/string.hpp>

#include <string>
#include <cstdint>

// ---- Simple finite-state machine ----

enum class RobotStateMachine : uint8_t
{
  IDLE = 0,
  READY,
  WALKING,
  MANIPULATING,
  ERROR,
  EMERGENCY_STOP
};

static const char * state_name(RobotStateMachine s)
{
  switch (s) {
    case RobotStateMachine::IDLE:            return "IDLE";
    case RobotStateMachine::READY:           return "READY";
    case RobotStateMachine::WALKING:         return "WALKING";
    case RobotStateMachine::MANIPULATING:    return "MANIPULATING";
    case RobotStateMachine::ERROR:           return "ERROR";
    case RobotStateMachine::EMERGENCY_STOP:  return "EMERGENCY_STOP";
    default:                                 return "UNKNOWN";
  }
}

class StrategyManagerNode : public rclcpp::Node
{
public:
  StrategyManagerNode()
  : Node("strategy_manager_node")
  {
    // Publishers → serial_bridge
    leg_cmd_pub_ = this->create_publisher<quadruped_interfaces::msg::LegCommand>(
      "leg_command", rclcpp::QoS(10));
    arm_pump_cmd_pub_ = this->create_publisher<quadruped_interfaces::msg::ArmPumpCommand>(
      "arm_pump_command", rclcpp::QoS(10));

    // Subscribers
    robot_state_sub_ = this->create_subscription<quadruped_interfaces::msg::RobotState>(
      "robot_state", rclcpp::QoS(10),
      [this](quadruped_interfaces::msg::RobotState::SharedPtr msg) {
        latest_robot_state_ = *msg;
      });

    // Vision results (placeholder topic — replace with actual vision output type)
    vision_sub_ = this->create_subscription<std_msgs::msg::String>(
      "vision_result", rclcpp::QoS(10),
      [this](std_msgs::msg::String::SharedPtr msg) {
        vision_callback(msg);
      });

    // 100 Hz control loop
    control_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(10),
      [this]() { control_loop(); });

    // ---- TODO: BehaviorTree.CPP integration point ----
    // bt_factory_ = std::make_unique<BT::BehaviorTreeFactory>();
    // bt_tree_     = bt_factory_->createTree("MainTree", blackboard_);

    RCLCPP_INFO(this->get_logger(), "Strategy Manager started. Initial state: IDLE");
  }

private:
  rclcpp::Publisher<quadruped_interfaces::msg::LegCommand>::SharedPtr leg_cmd_pub_;
  rclcpp::Publisher<quadruped_interfaces::msg::ArmPumpCommand>::SharedPtr arm_pump_cmd_pub_;
  rclcpp::Subscription<quadruped_interfaces::msg::RobotState>::SharedPtr robot_state_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr vision_sub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  RobotStateMachine state_{RobotStateMachine::IDLE};
  quadruped_interfaces::msg::RobotState latest_robot_state_;
  std::string latest_vision_result_;

  // ---- Callbacks ----

  void vision_callback(std_msgs::msg::String::SharedPtr msg)
  {
    latest_vision_result_ = msg->data;
    RCLCPP_DEBUG(this->get_logger(), "Vision update: %s", msg->data.c_str());
    on_vision_update(msg->data);
  }

  // ---- Main 100 Hz tick ----

  void control_loop()
  {
    // Check for emergency / error escalation from hardware first
    if (latest_robot_state_.robot_state == 4) {  // E-STOP
      transition_to(RobotStateMachine::EMERGENCY_STOP);
    } else if (latest_robot_state_.robot_state == 3) {  // Error reported by MCU
      transition_to(RobotStateMachine::ERROR);
    }

    switch (state_) {
      case RobotStateMachine::IDLE:
        idle_tick();
        break;
      case RobotStateMachine::READY:
        ready_tick();
        break;
      case RobotStateMachine::WALKING:
        walking_tick();
        break;
      case RobotStateMachine::MANIPULATING:
        manipulating_tick();
        break;
      case RobotStateMachine::ERROR:
        error_tick();
        break;
      case RobotStateMachine::EMERGENCY_STOP:
        estop_tick();
        break;
    }
  }

  // ---- Per-state tick functions ----

  void idle_tick()
  {
    // Wait for the MCU to report it's ready (robot_state == 0)
    if (latest_robot_state_.robot_state == 0) {
      transition_to(RobotStateMachine::READY);
    }
  }

  void ready_tick()
  {
    // Standing by for external commands (Nav2 goals, manual teleop, etc.)
    // No autonomous action — publish zero commands to hold position
  }

  void walking_tick()
  {
    // TODO: Execute walking gait.  This will eventually be driven by Nav2
    // cmd_vel → gait generator → LegCommand → serial_bridge.
  }

  void manipulating_tick()
  {
    // TODO: Execute arm/pump sequence triggered by vision detection.
    auto cmd = quadruped_interfaces::msg::ArmPumpCommand();
    // Fill in target positions here
    arm_pump_cmd_pub_->publish(cmd);
  }

  void error_tick()
  {
    // Attempt recovery or wait for human intervention
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
      "Robot in ERROR state (code %u): %s",
      latest_robot_state_.error_code, latest_robot_state_.error_msg.c_str());
  }

  void estop_tick()
  {
    // Publish zero commands, do nothing else
    auto leg_cmd = quadruped_interfaces::msg::LegCommand();
    auto arm_cmd = quadruped_interfaces::msg::ArmPumpCommand();
    leg_cmd.control_mode = 0;
    arm_cmd.pump_on = false;
    leg_cmd_pub_->publish(leg_cmd);
    arm_pump_cmd_pub_->publish(arm_cmd);
  }

  // ---- Vision event handler (called on each new vision result) ----

  void on_vision_update(const std::string & result)
  {
    // TODO: Parse detection result, decide whether to stop walking,
    // transition to MANIPULATING, etc.
    RCLCPP_DEBUG(this->get_logger(), "Processing vision result: %s", result.c_str());
  }

  // ---- State transition helper ----

  void transition_to(RobotStateMachine new_state)
  {
    if (new_state == state_) {
      return;
    }
    RCLCPP_INFO(this->get_logger(), "State: %s → %s",
                state_name(state_), state_name(new_state));
    state_ = new_state;
    // TODO: on_exit / on_enter hooks for each state
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StrategyManagerNode>());
  rclcpp::shutdown();
  return 0;
}
