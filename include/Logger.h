// MIT License
// Copyright (c) 2025 Andrew Kelleher
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
#define CLOG(level, ...) Logger::log(level, __FILE__, __func__, __LINE__, __VA_ARGS__)
#else
#define CLOG(level, ...) if (level == CRITICAL_LEVEL || level == ERROR_LEVEL) { Logger::log(level, __FILE__, __func__, __LINE__, __VA_ARGS__); }
#endif
#endif