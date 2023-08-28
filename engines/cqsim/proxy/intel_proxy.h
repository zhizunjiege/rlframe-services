#pragma once

#ifdef _WIN32
#include <Windows.h>
#include <atlbase.h>
#include <atlwin.h>
#else
#include <dlfcn.h>
#endif

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "csmodel_base.h"
#include "intel_proxy_export.h"

#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>
#include <json.hpp>

#include "protos/agent.grpc.pb.h"
#include "protos/agent.pb.h"
#include "protos/simenv.grpc.pb.h"
#include "protos/simenv.pb.h"
#include "protos/types.pb.h"

namespace agent = game::agent;
namespace simenv = game::simenv;
namespace types = game::types;

using json = nlohmann::json;

using CSValueVec = std::vector<std::any>;
using CSValueMap = std::unordered_map<std::string, std::any>;

extern "C" {
	INTEL_PROXY_EXPORT CSModelObject* CreateModelObject();
	INTEL_PROXY_EXPORT void DestroyMemory(void* mem, bool is_array);
}

// 智能代理
class IntelProxy : public CSModelObject {
public:
	// 初始化
	virtual bool Init(const std::unordered_map<std::string, std::any>& value)
		override;
	// 单步运算 time 应推进的步长(ms)
	virtual bool Tick(double time) override;
	// 获取输入参数
	virtual bool SetInput(const std::unordered_map<std::string, std::any>& value)
		override;
	// 对外部输出模型参数
	virtual std::unordered_map<std::string, std::any>* GetOutput() override;

public:
	~IntelProxy();

private:
	std::any DefaultValue(const std::string& type);

	std::any JsonToAny(const json& value, const std::string& type);

	types::SimParam AnyToMsg(const std::any& value, const std::string& type);

	std::any MsgToAny(const types::SimParam& value, const std::string& type);
private:
	// 模型动态库(dll/so)所在路径
	std::string library_dir_;

	// Proxy配置
	json configs_;

	// 参数类型定义
	std::unordered_map<std::string, std::unordered_map<std::string, std::string>> types_;
	// 输入输出类型
	std::unordered_map<std::string, std::unordered_map<std::string, std::string>> data_;
	// 数据路由配置
	std::unordered_map<std::string, std::vector<std::string>> routes_;

	// Agent服务的Stub,Stream,Context
	std::unordered_map<std::string, std::shared_ptr<agent::Agent::Stub>> stubs_;
	std::unordered_map<std::string, std::shared_ptr<grpc::ClientContext>> contexts_;
	std::unordered_map<std::string, std::shared_ptr<grpc::ClientReaderWriter<types::SimState, types::SimAction>>> streams_;
	// Simenv服务的Stub
	std::shared_ptr<simenv::Simenv::Stub > simenv_;

	// 仿真时长计时器
	double sim_duration_;
	double sim_times_;
	// 仿真步数计数器
	int32_t sim_step_ratio_;
	int32_t sim_steps_;

	// 仿真结束判断函数
#ifdef _WIN32
	HMODULE hmodule;
#else
	void* hmodule;
#endif
	// 结束判断函数
	typedef bool (*SimTermFunc)(const char* data, int32_t size);
	SimTermFunc sim_term_func_;

	// 输入输出缓存
	std::unordered_map<std::string, std::vector<CSValueMap>> inputs_;
	std::unordered_map<std::string, CSValueMap> outputs_;
};
