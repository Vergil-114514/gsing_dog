/**
 * serial_bridge_node — USB CDC serial bridge between ROS2 and the MCU
 *
 * Responsibilities:
 *   - Open & maintain a full-duplex serial link to the lower-level controller
 *   - Subscribe to LegCommand / ArmPumpCommand, pack them, and write to serial
 *   - Read incoming serial data at 50 Hz, unpack RobotState, and publish it
 *
 * Platform: Linux (termios). On Windows the serial I/O is stubbed out so the
 * package compiles for development, but no real serial traffic is processed.
 */

#ifdef __linux__
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#endif

#include <rclcpp/rclcpp.hpp>
#include <quadruped_interfaces/msg/leg_command.hpp>
#include <quadruped_interfaces/msg/arm_pump_command.hpp>
#include <quadruped_interfaces/msg/robot_state.hpp>

#include <string>
#include <vector>
#include <mutex>
#include <atomic>

class SerialBridgeNode : public rclcpp::Node
{
public:
  SerialBridgeNode()
  : Node("serial_bridge_node")
  {
    // Parameters
    this->declare_parameter("port", "/dev/ttyACM0");
    this->declare_parameter("baudrate", 115200);

    // Publishers
    robot_state_pub_ = this->create_publisher<quadruped_interfaces::msg::RobotState>(
      "robot_state", rclcpp::QoS(10));

    // Subscribers
    leg_cmd_sub_ = this->create_subscription<quadruped_interfaces::msg::LegCommand>(
      "leg_command", rclcpp::QoS(10),
      [this](quadruped_interfaces::msg::LegCommand::SharedPtr msg) {
        leg_cmd_callback(std::move(msg));
      });

    arm_pump_cmd_sub_ = this->create_subscription<quadruped_interfaces::msg::ArmPumpCommand>(
      "arm_pump_command", rclcpp::QoS(10),
      [this](quadruped_interfaces::msg::ArmPumpCommand::SharedPtr msg) {
        arm_pump_cmd_callback(std::move(msg));
      });

    // 50 Hz read timer
    read_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(20),
      [this]() { read_serial_callback(); });

    // Open serial port
    std::string port = this->get_parameter("port").as_string();
    int baudrate = this->get_parameter("baudrate").as_int();

    if (open_serial(port, baudrate)) {
      RCLCPP_INFO(this->get_logger(), "Serial bridge online. Port: %s, Baud: %d",
                  port.c_str(), baudrate);
    }
  }

  ~SerialBridgeNode() override
  {
    running_ = false;
#ifdef __linux__
    if (serial_fd_ >= 0) {
      close(serial_fd_);
    }
#endif
  }

private:
  // ---- ROS2 handles ----
  rclcpp::Publisher<quadruped_interfaces::msg::RobotState>::SharedPtr robot_state_pub_;
  rclcpp::Subscription<quadruped_interfaces::msg::LegCommand>::SharedPtr leg_cmd_sub_;
  rclcpp::Subscription<quadruped_interfaces::msg::ArmPumpCommand>::SharedPtr arm_pump_cmd_sub_;
  rclcpp::TimerBase::SharedPtr read_timer_;

  // ---- Serial I/O state ----
  int serial_fd_ = -1;
  std::atomic<bool> running_{true};
  std::mutex write_mutex_;

  // ---- Incoming command callbacks ----

  void leg_cmd_callback(quadruped_interfaces::msg::LegCommand::SharedPtr msg)
  {
    auto packet = pack_leg_command(msg);
    write_serial(packet);
  }

  void arm_pump_cmd_callback(quadruped_interfaces::msg::ArmPumpCommand::SharedPtr msg)
  {
    auto packet = pack_arm_pump_command(msg);
    write_serial(packet);
  }

  // ---- 50 Hz serial read ----

  void read_serial_callback()
  {
#ifdef __linux__
    if (serial_fd_ < 0) {
      return;
    }

    auto raw = read_serial(1024);
    if (!raw.empty()) {
      rx_buffer_.insert(rx_buffer_.end(), raw.begin(), raw.end());
      // Attempt to extract complete frames
      while (auto state = try_parse_frame()) {
        state->header.stamp = this->now();
        state->header.frame_id = "base_link";
        robot_state_pub_->publish(*state);
      }
    }
#else
    // Windows stub — publish empty state for development
    auto state = quadruped_interfaces::msg::RobotState();
    state.header.stamp = this->now();
    state.header.frame_id = "base_link";
    robot_state_pub_->publish(state);
#endif
  }

  // ---- Data packing (PLACEHOLDERS — replace with real protocol) ----

  std::vector<uint8_t> pack_leg_command(
    const quadruped_interfaces::msg::LegCommand::SharedPtr & msg)
  {
    // TODO: Implement actual binary protocol packing
    std::vector<uint8_t> packet;
    packet.reserve(2 + 12 * 4 + 12 * 4 + 1 + 1);  // header + joints + ctrl + footer

    packet.push_back(0xAA);  // frame header
    packet.push_back(0x01);  // message type: leg command

    for (size_t i = 0; i < 12; ++i) {
      auto raw = reinterpret_cast<const uint8_t *>(&msg->joint_positions[i]);
      packet.insert(packet.end(), raw, raw + sizeof(float));
    }
    for (size_t i = 0; i < 12; ++i) {
      auto raw = reinterpret_cast<const uint8_t *>(&msg->joint_velocities[i]);
      packet.insert(packet.end(), raw, raw + sizeof(float));
    }
    packet.push_back(msg->control_mode);
    packet.push_back(0x55);  // frame footer

    return packet;
  }

  std::vector<uint8_t> pack_arm_pump_command(
    const quadruped_interfaces::msg::ArmPumpCommand::SharedPtr & msg)
  {
    // TODO: Implement actual binary protocol packing
    std::vector<uint8_t> packet;
    packet.reserve(2 + 6 * 4 + 1 + 4 + 1);

    packet.push_back(0xAA);
    packet.push_back(0x02);  // message type: arm + pump command

    for (size_t i = 0; i < 6; ++i) {
      auto raw = reinterpret_cast<const uint8_t *>(&msg->joint_positions[i]);
      packet.insert(packet.end(), raw, raw + sizeof(float));
    }
    packet.push_back(msg->pump_on ? 0x01 : 0x00);
    {
      auto raw = reinterpret_cast<const uint8_t *>(&msg->pump_pressure);
      packet.insert(packet.end(), raw, raw + sizeof(float));
    }
    packet.push_back(0x55);

    return packet;
  }

  // ---- Frame parsing (PLACEHOLDER — replace with real protocol) ----

  std::optional<quadruped_interfaces::msg::RobotState> try_parse_frame()
  {
    // TODO: Implement actual binary protocol parsing
    // Search rx_buffer_ for a complete frame, extract & remove it
    return std::nullopt;
  }

  // ---- Platform-specific serial I/O ----

  bool open_serial(const std::string & port, int baudrate)
  {
#ifdef __linux__
    serial_fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serial_fd_ < 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open %s: %s", port.c_str(), strerror(errno));
      return false;
    }

    struct termios tty;
    std::memset(&tty, 0, sizeof(tty));
    if (tcgetattr(serial_fd_, &tty) != 0) {
      RCLCPP_ERROR(this->get_logger(), "tcgetattr failed: %s", strerror(errno));
      close(serial_fd_);
      serial_fd_ = -1;
      return false;
    }

    speed_t speed = B115200;
    switch (baudrate) {
      case 9600:   speed = B9600;   break;
      case 57600:  speed = B57600;  break;
      case 115200: speed = B115200; break;
      case 230400: speed = B230400; break;
      case 921600: speed = B921600; break;
      default:
        RCLCPP_WARN(this->get_logger(), "Unsupported baudrate %d, falling back to 115200", baudrate);
        speed = B115200;
        break;
    }
    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_oflag &= ~OPOST;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(serial_fd_, TCSANOW, &tty) != 0) {
      RCLCPP_ERROR(this->get_logger(), "tcsetattr failed: %s", strerror(errno));
      close(serial_fd_);
      serial_fd_ = -1;
      return false;
    }

    // Flush any stale data
    tcflush(serial_fd_, TCIOFLUSH);
    return true;
#else
    RCLCPP_WARN(this->get_logger(),
      "Serial I/O is not supported on this platform. "
      "The node will run but publish empty RobotState messages.");
    return false;
#endif
  }

  void write_serial(const std::vector<uint8_t> & data)
  {
#ifdef __linux__
    if (serial_fd_ < 0) {
      RCLCPP_WARN_ONCE(this->get_logger(), "Serial port not open; dropping write");
      return;
    }
    std::lock_guard<std::mutex> lock(write_mutex_);
    ssize_t written = ::write(serial_fd_, data.data(), data.size());
    if (written < 0) {
      RCLCPP_ERROR(this->get_logger(), "Serial write error: %s", strerror(errno));
    }
#else
    (void)data;
#endif
  }

  std::vector<uint8_t> read_serial(size_t max_bytes)
  {
#ifdef __linux__
    std::vector<uint8_t> buf(max_bytes);
    ssize_t n = ::read(serial_fd_, buf.data(), max_bytes);
    if (n > 0) {
      buf.resize(static_cast<size_t>(n));
      return buf;
    }
    return {};
#else
    (void)max_bytes;
    return {};
#endif
  }

  std::vector<uint8_t> rx_buffer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SerialBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
