#include "serialize.hpp"

extern "C" __declspec(dllexport) bool __sim_term_func__(const char *data, int32_t size)
{
  std::string str(data, size);
  serialize::Deserializer deserializer;
  auto states = deserializer.Deserialize(str);
  return func(states);
}
