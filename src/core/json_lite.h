#pragma once

#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pahnts {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::map<std::string, JsonValue> obj_val;

    bool is_null() const { return type == Type::Null; }
    bool as_bool(bool def = false) const { return type == Type::Bool ? bool_val : def; }
    double as_double(double def = 0.0) const { return type == Type::Number ? num_val : def; }
    float as_float(float def = 0.0f) const { return static_cast<float>(as_double(def)); }
    int as_int(int def = 0) const { return type == Type::Number ? static_cast<int>(num_val) : def; }
    uint64_t as_u64(uint64_t def = 0) const { return type == Type::Number ? static_cast<uint64_t>(num_val) : def; }
    const std::string& as_string() const { return str_val; }

    bool has(const std::string& k) const {
        return type == Type::Object && obj_val.find(k) != obj_val.end();
    }
    const JsonValue& operator[](const std::string& k) const {
        static const JsonValue kNull;
        if (type != Type::Object) return kNull;
        auto it = obj_val.find(k);
        return it == obj_val.end() ? kNull : it->second;
    }
    const JsonValue& operator[](size_t idx) const {
        static const JsonValue kNull;
        if (type != Type::Array || idx >= arr_val.size()) return kNull;
        return arr_val[idx];
    }
    size_t size() const {
        if (type == Type::Array) return arr_val.size();
        if (type == Type::Object) return obj_val.size();
        return 0;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& src) : s_(src), p_(0) {}

    static JsonValue parse_file(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) throw std::runtime_error("Failed to open JSON file: " + path);
        std::ostringstream oss;
        oss << ifs.rdbuf();
        std::string content = oss.str();
        if (content.size() >= 3 &&
            static_cast<unsigned char>(content[0]) == 0xEF &&
            static_cast<unsigned char>(content[1]) == 0xBB &&
            static_cast<unsigned char>(content[2]) == 0xBF) {
            content.erase(0, 3);
        }
        try {
            JsonParser parser(content);
            return parser.parse_value();
        } catch (const std::exception& ex) {
            throw std::runtime_error("JSON parse error in " + path + ": " + ex.what());
        }
    }

    JsonValue parse_value() {
        skip_ws();
        if (p_ >= s_.size()) throw std::runtime_error("Unexpected end of JSON");
        char c = s_[p_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            JsonValue v;
            v.type = JsonValue::Type::String;
            v.str_val = parse_string();
            return v;
        }
        if (c == 't') { expect("true"); JsonValue v; v.type = JsonValue::Type::Bool; v.bool_val = true; return v; }
        if (c == 'f') { expect("false"); JsonValue v; v.type = JsonValue::Type::Bool; v.bool_val = false; return v; }
        if (c == 'n') { expect("null"); return JsonValue{}; }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        throw std::runtime_error(std::string("Unexpected JSON char: ") + c);
    }

private:
    const std::string& s_;
    size_t p_;

    void skip_ws() {
        while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
    }

    void expect(const char* lit) {
        for (const char* q = lit; *q; ++q, ++p_) {
            if (p_ >= s_.size() || s_[p_] != *q) {
                throw std::runtime_error(std::string("Expected ") + lit);
            }
        }
    }

    std::string parse_string() {
        ++p_; // skip '"'
        std::string out;
        while (p_ < s_.size()) {
            char c = s_[p_++];
            if (c == '"') return out;
            if (c == '\\' && p_ < s_.size()) {
                char e = s_[p_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u':
                        if (p_ + 4 <= s_.size()) {
                            unsigned int code = std::stoul(s_.substr(p_, 4), nullptr, 16);
                            p_ += 4;
                            if (code < 0x80) out.push_back(static_cast<char>(code));
                            else out.push_back('?');
                        }
                        break;
                    default: out.push_back(e); break;
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }

    JsonValue parse_number() {
        size_t start = p_;
        if (s_[p_] == '-') ++p_;
        while (p_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[p_])) ||
               s_[p_] == '.' || s_[p_] == 'e' || s_[p_] == 'E' || s_[p_] == '+' || s_[p_] == '-')) {
            ++p_;
        }
        JsonValue v;
        v.type = JsonValue::Type::Number;
        v.num_val = std::stod(s_.substr(start, p_ - start));
        return v;
    }

    JsonValue parse_array() {
        ++p_; // '['
        JsonValue v;
        v.type = JsonValue::Type::Array;
        skip_ws();
        if (p_ < s_.size() && s_[p_] == ']') { ++p_; return v; }
        while (true) {
            v.arr_val.push_back(parse_value());
            skip_ws();
            if (p_ < s_.size() && s_[p_] == ',') { ++p_; continue; }
            if (p_ < s_.size() && s_[p_] == ']') { ++p_; break; }
            throw std::runtime_error("Malformed JSON array");
        }
        return v;
    }

    JsonValue parse_object() {
        ++p_; // '{'
        JsonValue v;
        v.type = JsonValue::Type::Object;
        skip_ws();
        if (p_ < s_.size() && s_[p_] == '}') { ++p_; return v; }
        while (true) {
            skip_ws();
            if (p_ >= s_.size() || s_[p_] != '"') throw std::runtime_error("Expected JSON key");
            std::string key = parse_string();
            skip_ws();
            if (p_ >= s_.size() || s_[p_] != ':') throw std::runtime_error("Expected ':'");
            ++p_;
            v.obj_val.emplace(std::move(key), parse_value());
            skip_ws();
            if (p_ < s_.size() && s_[p_] == ',') { ++p_; continue; }
            if (p_ < s_.size() && s_[p_] == '}') { ++p_; break; }
            throw std::runtime_error("Malformed JSON object");
        }
        return v;
    }
};

} // namespace pahnts
