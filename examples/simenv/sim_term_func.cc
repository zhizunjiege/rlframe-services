#include <any>
#include <string>
#include <unordered_map>
#include <vector>

using CSValueMap = std::unordered_map<std::string, std::any>;

extern "C" __declspec(dllexport) bool func(std::unordered_map<std::string, std::vector<CSValueMap>> &states)
{
  return false;
}
