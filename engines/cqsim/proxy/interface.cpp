#include "serialize.hpp"

bool func(std::unordered_map<std::string, std::vector<std::unordered_map<std::string, std::any>>> &states);

extern "C" __declspec(dllexport) bool __sim_term_func__(const char *data, int32_t size)
{
  std::string str(data, size);
  serialize::Deserializer deserializer;
  auto states = deserializer.Deserialize(str);
  return func(states);
}
