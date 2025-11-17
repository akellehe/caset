#include "Logger.h"

#include <chrono>
#include <iomanip>
#include <iostream>

std::string Logger::getTime() {
  auto now = std::chrono::system_clock::now();
  auto as_time_t = std::chrono::system_clock::to_time_t(now);
  auto as_tm = std::localtime(&as_time_t);
  std::stringstream ss;
  ss << std::put_time(as_tm, "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

short int Logger::getLevel() {
  if (Logger::LEVEL) {
    return Logger::LEVEL.value();
  }

  char *log_level = std::getenv("LOG_LEVEL");

  if (log_level) {
    short int ll = std::stoi(log_level);
    if (ll == DEBUG_LEVEL) {
      Logger::LEVEL = DEBUG_LEVEL;
    } else if (ll == INFO_LEVEL) {
      Logger::LEVEL = INFO_LEVEL;
    } else if (ll == WARN_LEVEL) {
      Logger::LEVEL = WARN_LEVEL;
    } else if (ll == ERROR_LEVEL) {
      Logger::LEVEL = ERROR_LEVEL;
    } else if (ll == CRITICAL_LEVEL) {
      Logger::LEVEL = CRITICAL_LEVEL;
    } else {
      Logger::LEVEL = INFO_LEVEL;
    }
  } else {
    Logger::LEVEL = INFO_LEVEL;
  }

  return Logger::LEVEL.value();
}

std::string Logger::nameLevel(short int level) {
  if (level == DEBUG_LEVEL) {
    return "DEBUG";
  } else if (level == INFO_LEVEL) {
    return "INFO";
  } else if (level == WARN_LEVEL) {
    return "WARN";
  } else if (level == ERROR_LEVEL) {
    return "ERROR";
  } else if (level == CRITICAL_LEVEL) {
    return "CRITICAL";
  } else {
    return "UNKNOWN";
  }

}

void Logger::emit(short int level,
                  const std::string &filename,
                  std::string func,
                  const int lineno,
                  std::string &message) {
  std::cout
      << Logger::getTime()
      << " - "
      << Logger::nameLevel(level)
      << " - "
      << Logger::makeRelative(filename, SOURCES_ROOT)
      << ":L"
      << lineno
      << ":" << func << "()"
      << ": "
      << message
      << std::endl;
}

std::string Logger::makeRelative(const std::string &absolute, const std::string &root) {
  size_t pos = absolute.find(root);
  if (pos != std::string::npos) {
    return absolute.substr(root.size() + 1, absolute.size() - pos);
  }
  return absolute;
}

std::optional<short int> Logger::LEVEL = Logger::getLevel();
