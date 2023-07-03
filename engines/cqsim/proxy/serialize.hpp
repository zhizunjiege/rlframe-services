#pragma once
#include <any>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace serialize
{
	using CSValueMap = std::unordered_map<std::string, std::any>;
	using CSValueVec = std::vector<std::any>;

	enum DataType
	{
		kVoid = 0,
		kBool = 1,
		kInt8 = 2,
		kUint8 = 3,
		kInt16 = 4,
		kUint16 = 5,
		kInt32 = 6,
		kUint32 = 7,
		kInt64 = 8,
		kUint64 = 9,
		kFloat32 = 10,
		kFloat64 = 11,
		kFloat128 = 12,
		kString = 13,
		kArray = 14,
		kStruct = 15,
	};

	class Serializer
	{
	private:
		std::ostringstream os_;

	public:
		Serializer() : os_() {}

		std::string Serialize(const std::unordered_map<std::string, std::vector<std::unordered_map<std::string, std::any>>> &data)
		{
			size_t size1 = data.size();
			*this << size1;
			for (auto &it1 : data)
			{
				*this << it1.first;
				size_t size2 = it1.second.size();
				*this << size2;
				for (auto &it2 : it1.second)
				{
					size_t size3 = it2.size();
					*this << size3;
					for (auto &it3 : it2)
					{
						*this << it3.first;
						*this << it3.second;
					}
				}
			}
			return os_.str();
		};

		template <DataType TypeIdentity, typename SourceType, typename TargetType>
		bool DefAnyType(const std::type_info &type, const std::any &data)
		{
			if (type == typeid(SourceType))
			{
				uint8_t id = TypeIdentity;
				TargetType val = std::any_cast<SourceType>(data);
				*this << id << val;
				return true;
			}
			else
			{
				return false;
			}
		}

		Serializer &operator<<(const std::any &data)
		{
			auto &type = data.type();
			if (DefAnyType<kStruct, CSValueMap, CSValueMap>(type, data) ||
				DefAnyType<kArray, CSValueVec, CSValueVec>(type, data) ||
				DefAnyType<kFloat64, double, double>(type, data) ||
				DefAnyType<kUint64, uint64_t, uint64_t>(type, data) ||
				DefAnyType<kUint16, uint16_t, uint16_t>(type, data) ||
				DefAnyType<kBool, bool, uint8_t>(type, data) ||
				DefAnyType<kInt32, int32_t, int32_t>(type, data) ||
				DefAnyType<kString, std::string, std::string>(type, data) ||
				DefAnyType<kFloat32, float, float>(type, data) ||
				DefAnyType<kFloat128, long double, double>(type, data) ||
				DefAnyType<kUint32, uint32_t, uint32_t>(type, data) ||
				DefAnyType<kUint8, uint8_t, uint8_t>(type, data) ||
				DefAnyType<kInt64, int64_t, int64_t>(type, data) ||
				DefAnyType<kInt16, int16_t, int16_t>(type, data) ||
				DefAnyType<kInt8, int8_t, int8_t>(type, data))
			{
				return *this;
			}
			else
			{
				throw std::string("Unsupported type!");
			}
		}

		Serializer &operator<<(const CSValueMap &data)
		{
			uint16_t size = data.size();
			*this << size;
			for (auto &it : data)
			{
				*this << it.first;
				*this << it.second;
			}
			return *this;
		};

		Serializer &operator<<(const CSValueVec &data)
		{
			uint16_t size = data.size();
			*this << size;
			for (auto &it : data)
			{
				*this << it;
			}
			return *this;
		};

		Serializer &operator<<(const std::string &data)
		{
			uint16_t size = data.size();
			*this << size;
			os_.write(data.data(), size);
			return *this;
		}

		template <typename BasicType>
		Serializer &operator<<(const BasicType &data)
		{
			os_.write(reinterpret_cast<const char *>(&data), sizeof(BasicType));
			return *this;
		}
	};

	class Deserializer
	{
	private:
		std::istringstream is_;

	public:
		Deserializer() : is_() {}

		std::unordered_map<std::string, std::vector<std::unordered_map<std::string, std::any>>> Deserialize(const std::string &data)
		{
			is_.str(data);
			std::unordered_map<std::string, std::vector<std::unordered_map<std::string, std::any>>> ret;
			size_t size1;
			*this >> size1;
			for (auto it1 = 0; it1 < size1; it1++)
			{
				std::string key1;
				*this >> key1;
				std::vector<std::unordered_map<std::string, std::any>> vec;
				size_t size2;
				*this >> size2;
				for (auto it2 = 0; it2 < size2; it2++)
				{
					std::unordered_map<std::string, std::any> map;
					size_t size3;
					*this >> size3;
					for (auto it3 = 0; it3 < size3; it3++)
					{
						std::string key3;
						std::any value3;
						*this >> key3 >> value3;
						map.emplace(key3, value3);
					}
					vec.emplace_back(map);
				}
				ret.emplace(key1, vec);
			}
			return ret;
		};

		template <DataType TypeIdentity, typename SourceType, typename TargetType>
		bool DefAnyType(uint8_t &type, std::any &data)
		{
			if (type == (uint8_t)TypeIdentity)
			{
				TargetType tmp;
				*this >> tmp;
				data = (SourceType)tmp;
				return true;
			}
			else
			{
				return false;
			}
		}

		Deserializer &operator>>(std::any &data)
		{
			uint8_t type;
			*this >> type;
			if (DefAnyType<kStruct, CSValueMap, CSValueMap>(type, data) ||
				DefAnyType<kArray, CSValueVec, CSValueVec>(type, data) ||
				DefAnyType<kFloat64, double, double>(type, data) ||
				DefAnyType<kUint64, uint64_t, uint64_t>(type, data) ||
				DefAnyType<kUint16, uint16_t, uint16_t>(type, data) ||
				DefAnyType<kBool, bool, uint8_t>(type, data) ||
				DefAnyType<kInt32, int32_t, int32_t>(type, data) ||
				DefAnyType<kString, std::string, std::string>(type, data) ||
				DefAnyType<kFloat32, float, float>(type, data) ||
				DefAnyType<kFloat128, long double, double>(type, data) ||
				DefAnyType<kUint32, uint32_t, uint32_t>(type, data) ||
				DefAnyType<kUint8, uint8_t, uint8_t>(type, data) ||
				DefAnyType<kInt64, int64_t, int64_t>(type, data) ||
				DefAnyType<kInt16, int16_t, int16_t>(type, data) ||
				DefAnyType<kInt8, int8_t, int8_t>(type, data))
			{
				return *this;
			}
			else
			{
				throw std::string("Unsupported type!");
			}
		}

		Deserializer &operator>>(CSValueMap &data)
		{
			uint16_t size;
			*this >> size;
			for (auto it = 0; it < size; it++)
			{
				std::string tmp_key;
				std::any tmp_value;
				*this >> tmp_key;
				*this >> tmp_value;
				data.emplace(tmp_key, tmp_value);
			}
			return *this;
		};

		Deserializer &operator>>(CSValueVec &data)
		{
			uint16_t size;
			*this >> size;
			for (auto it = 0; it < size; it++)
			{
				std::any tmp;
				*this >> tmp;
				data.emplace_back(tmp);
			}
			return *this;
		};

		Deserializer &operator>>(std::string &data)
		{
			uint16_t size;
			*this >> size;
			char *tmp = new char[size];
			is_.read(tmp, size);
			data = std::string(tmp, size);
			delete[] tmp;
			return *this;
		}

		template <typename BasicType>
		Deserializer &operator>>(BasicType &data)
		{
			is_.read(reinterpret_cast<char *>(&data), sizeof(BasicType));
			return *this;
		}
	};
}
