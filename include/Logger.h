#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <cstdlib>
#include <optional>
#include <sstream>

const short int DEBUG_LEVEL = 10;
const short int INFO_LEVEL = 20;
const short int WARN_LEVEL = 30;
const short int ERROR_LEVEL = 40;
const short int CRITICAL_LEVEL = 50;


class Logger {

 public:
  static std::optional<short int> LEVEL;
  static std::string getTime();
  static short int getLevel();
  static std::string nameLevel(short int level);

  static void emit(short int level,
                   const std::string &filename,
                   std::string func,
                   const int lineno,
                   std::string &message);

  static std::string makeRelative(const std::string &absolute, const std::string &root);

  template<typename... Args>
  static void log(const short int level, const std::string &filename, std::string func, int lineno, Args &&... args) {
    if (level >= Logger::getLevel()) {
      std::ostringstream message;
      (message << ... << args);
      std::string as_str = message.str();
      std::string relative = Logger::makeRelative(filename, SOURCES_ROOT);
      emit(level, relative, func, lineno, as_str);
    }
  }
};
#ifdef VERBOSE
#define LOG(level, ...) Logger::log(level, __FILE__, __func__, __LINE__, __VA_ARGS__)
#else
#define LOG(level, ...) if (level == CRITICAL_LEVEL || level == ERROR_LEVEL) { Logger::log(level, __FILE__, __func__, __LINE__, __VA_ARGS__); }
#endif
#endif