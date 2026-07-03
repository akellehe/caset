// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_OBSERVABLES_RECORD_H
#define TESSERA_OBSERVABLES_RECORD_H

#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace tessera::observables {

/// # Record
///
/// A JSON-able observable record — the C++ home of the Python framework's
/// nested `dict` of `float / int / bool / str / None / list / dict` leaves
/// (#593). Every emergent-proton observable's `record()` returns one; the
/// binding layer turns it into a Python `dict`. Keeping the record type in
/// pybind-free `tessera_core` lets the GAUGE/RELABEL gates traverse it in C++
/// (`reportDelta`).
///
/// The propagation discipline (#580) lives at the reporting layer: a channel
/// is real BY CONSTRUCTION or it carries both parts explicitly — complex
/// values enter through `splitComplex`, which stores the two real leaves
/// `{name}_re` / `{name}_im`; nothing is ever silently `.real`-ed.
class Record {
  public:
    enum class Type { Null, Bool, Int, Double, String, List, Map };
    using List = std::vector<Record>;
    using Map = std::map<std::string, Record>;

    Record() noexcept : type_(Type::Null) {}
    Record(std::nullptr_t) noexcept : type_(Type::Null) {}  // NOLINT
    Record(bool value) noexcept : type_(Type::Bool), bool_(value) {}  // NOLINT
    // Integer leaves (`Int`). Explicit per-type overloads rather than one
    // `enable_if_t` template so the Breathe/doxygen C++ parser can render the
    // header; the set covers every fundamental integer type (so `int`,
    // `std::size_t`, `std::int64_t`, `std::uint64_t`, ... all bind without
    // ambiguity on LP64).
    Record(int value) noexcept : type_(Type::Int), int_(value) {}  // NOLINT
    Record(unsigned int value) noexcept  // NOLINT
        : type_(Type::Int), int_(value) {}
    Record(long value) noexcept : type_(Type::Int), int_(value) {}  // NOLINT
    Record(unsigned long value) noexcept  // NOLINT
        : type_(Type::Int), int_(static_cast<std::int64_t>(value)) {}
    Record(long long value) noexcept  // NOLINT
        : type_(Type::Int), int_(value) {}
    Record(unsigned long long value) noexcept  // NOLINT
        : type_(Type::Int), int_(static_cast<std::int64_t>(value)) {}
    // Floating-point leaves (`Double`; NaN / inf preserved).
    Record(double value) noexcept  // NOLINT
        : type_(Type::Double), double_(value) {}
    Record(float value) noexcept  // NOLINT
        : type_(Type::Double), double_(static_cast<double>(value)) {}
    Record(const char *value) : type_(Type::String), string_(value) {}  // NOLINT
    Record(std::string value)  // NOLINT
        : type_(Type::String), string_(std::move(value)) {}
    Record(List value) : type_(Type::List), list_(std::move(value)) {}  // NOLINT
    Record(Map value) : type_(Type::Map), map_(std::move(value)) {}  // NOLINT

    [[nodiscard]] Type type() const noexcept { return type_; }
    [[nodiscard]] bool isNull() const noexcept { return type_ == Type::Null; }

    [[nodiscard]] bool asBool() const { return bool_; }
    [[nodiscard]] std::int64_t asInt() const { return int_; }
    [[nodiscard]] double asDouble() const { return double_; }
    [[nodiscard]] const std::string &asString() const { return string_; }
    [[nodiscard]] const List &asList() const { return list_; }
    [[nodiscard]] const Map &asMap() const { return map_; }

    /// The two explicit JSON-able leaves `{name}_re` / `{name}_im` of a
    /// complex scalar — the one naming convention for complex record channels
    /// (#580: the imaginary part is real physics and is always carried).
    /// Merges them into `into`.
    static void splitComplex(Map &into, const std::string &name,
                             std::complex<double> value);
    /// `splitComplex` for a sequence: `{name}_re` / `{name}_im` become lists.
    static void splitComplex(Map &into, const std::string &name,
                             const std::vector<std::complex<double>> &values);

    /// The max absolute difference over every numeric leaf of two records —
    /// the C_ij chamber readout's every-channel gate metric (ported verbatim
    /// from the Python `report_delta`):
    ///
    ///   * maps must have identical keys (a shape mismatch throws);
    ///   * lists must have identical lengths (else `inf`);
    ///   * strings and nulls must be equal (else `inf` — a changed status IS a
    ///     flagged channel);
    ///   * bools compare as 0/1; numbers as `|a - b|`; two NaNs agree
    ///     (delta 0 — NaN is a legitimate reported value, e.g. a
    ///     not-applicable reading), a NaN against a number is `inf`.
    ///
    /// @throws std::invalid_argument when two maps have different key sets.
    [[nodiscard]] static double reportDelta(const Record &a, const Record &b);

  private:
    Type type_;
    bool bool_ = false;
    std::int64_t int_ = 0;
    double double_ = 0.0;
    std::string string_;
    List list_;
    Map map_;
};

}  // namespace tessera::observables

#endif  // TESSERA_OBSERVABLES_RECORD_H
