#include <any>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

/*
 * This function is called in every simulation step with a map of states, where the key is the name of the model and the value is a vector of entities.
 * Each entity in the vector represents a sets of output params, and the key is the name of the param and the value is wrapped in std::any.
 * The function should return `true` if the simulation should be terminated.
 */
bool func(std::unordered_map<std::string, std::vector<std::unordered_map<std::string, std::any>>> &states)
{
  return false;
}
