#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>

#include "serialize.hpp"

#include "intel_proxy.h"

bool IntelProxy::Init(const std::unordered_map<std::string, std::any>& value) {
#ifdef _WIN32
	HMODULE module_instance = _AtlBaseModule.GetModuleInstance();
	char dll_path[MAX_PATH] = { 0 };
	GetModuleFileNameA(module_instance, dll_path, _countof(dll_path));
	char drive[_MAX_DRIVE];
	char dir[_MAX_DIR];
	char fname[_MAX_FNAME];
	char ext[_MAX_EXT];
	_splitpath_s(dll_path, drive, dir, fname, ext);
	library_dir_ = drive + std::string(dir) + "\\";
#else
	Dl_info dl_info;
	CSModelObject* (*p)() = &CreateModelObject;
	if (0 != dladdr((void*)(p), &dl_info)) {
		library_dir_ = std::string(dl_info.dli_fname);
		library_dir_ = library_dir_.substr(0, library_dir_.find_last_of('/'));
		library_dir_ += "/";
	}
#endif

	std::ifstream fi(library_dir_ + "configs.json");
	configs_ = json::parse(fi);
	fi.close();

	for (auto& it1 : configs_["types"].items()) {
		auto type_name = it1.key();
		auto type_fields = it1.value();
		std::unordered_map<std::string, std::string> type_struct;
		for (auto& it2 : type_fields.items()) {
			auto field_name = it2.key();
			auto field_type = it2.value().get<std::string>();
			type_struct.emplace(field_name, field_type);
		}
		types_.emplace(type_name, type_struct);
	}

	for (auto& it1 : configs_["data"].items()) {
		auto model_name = it1.key();
		auto model_config = it1.value();
		std::unordered_map<std::string, std::string> model_params;
		CSValueMap model_inputs;
		for (auto& it2 : model_config["inputs"].items()) {
			auto input_name = it2.key();
			auto input_config = it2.value();
			std::string param_type;
			std::any param_value;
			if (input_config.type() == json::value_t::object) {
				param_type = input_config["type"].get<std::string>();
				param_value = JsonToAny(input_config["value"], param_type);
			}
			else {
				param_type = input_config.get<std::string>();
				param_value = DefaultValue(param_type);
			}
			model_params.emplace(input_name, param_type);
			model_inputs.emplace(input_name, param_value);
		}
		outputs_.emplace(model_name, model_inputs);
		std::vector<CSValueMap> model_outputs;
		for (auto& it2 : model_config["outputs"].items()) {
			auto output_name = it2.key();
			auto output_config = it2.value();
			std::string param_type;
			if (output_config.type() == json::value_t::object) {
				param_type = output_config["type"].get<std::string>();
			}
			else {
				param_type = output_config.get<std::string>();
			}
			model_params.emplace(output_name, param_type);
		}
		inputs_.emplace(model_name, model_outputs);
		data_.emplace(model_name, model_params);
	}

	for (auto& it1 : configs_["routes"].items()) {
		auto agent_addr = it1.key();
		auto agent_models = it1.value();
		routes_.emplace(agent_addr, std::vector<std::string>());
		for (auto& agent_model : agent_models) {
			routes_[agent_addr].emplace_back(agent_model.get<std::string>());
		}
		auto agent_channel = grpc::CreateChannel(agent_addr, grpc::InsecureChannelCredentials());
		std::shared_ptr<agent::Agent::Stub> agent_stub = agent::Agent::NewStub(agent_channel);
		stubs_.emplace(agent_addr, agent_stub);
		std::shared_ptr<grpc::ClientContext> agent_context = std::make_shared<grpc::ClientContext>();
		contexts_.emplace(agent_addr, agent_context);
		std::shared_ptr<grpc::ClientReaderWriter<types::SimState, types::SimAction>> agent_stream = agent_stub->GetAction(agent_context.get());
		streams_.emplace(agent_addr, agent_stream);
	}

	sim_duration_ = configs_["sim_duration"].get<double>() * 1000;
	sim_times_ = 0;
	sim_step_ratio_ = configs_["sim_step_ratio"].get<int32_t>();
	sim_steps_ = 0;

	auto simenv_addr = configs_["simenv_addr"].get<std::string>();
	auto simenv_channel = grpc::CreateChannel(simenv_addr, grpc::InsecureChannelCredentials());
	simenv_ = simenv::Simenv::NewStub(simenv_channel);

#ifdef _WIN32
	std::string lib_name = "sim_term_func.dll";
#else
	std::string lib_name = "libsim_term_func.so";
#endif
	std::string lib_path = library_dir_ + lib_name;
#ifdef _WIN32
	hmodule = LoadLibraryExA(lib_path.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
#else
	hmodule = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
#endif
	if (!hmodule) {
		WriteLog("Can not load " + lib_name, 5);
		return false;
	}
#ifdef _WIN32
	sim_term_func_ = (SimTermFunc)GetProcAddress(hmodule, "__sim_term_func__");
#else
	sim_term_func_ = (SimTermFunc)dlsym(hmodule, "__sim_term_func__");
#endif
	if (!sim_term_func_) {
		WriteLog("Can not find function with name `__sim_term_func__` in " + lib_name, 5);
		return false;
	}

	state_ = CSInstanceState::IS_INITIALIZED;

	WriteLog("intel_proxy model Init", 0);
	return true;
}

bool IntelProxy::Tick(double time) {
  if (state_ == CSInstanceState::IS_DESTROYED) {
		return true;
	}

	sim_times_ += time;
	if (sim_steps_ % sim_step_ratio_ != 0)
	{
		return true;
	}

	for (auto& it : outputs_)
	{
		it.second.clear();
	}

	try {
		serialize::Serializer serializer;
		auto data = serializer.Serialize(inputs_);

		bool terminated = sim_term_func_((const char*)data.data(), (int32_t)data.size());
		bool truncated = sim_times_ + time * ((double)sim_step_ratio_ - 1) + 0.001 >= sim_duration_;

		for (auto& [agent_addr, agent_models] : routes_) {
			types::SimState req;
			auto states = req.mutable_states();
			for (auto& model_name : agent_models) {
				(*states)[model_name] = types::SimModel();
				for (auto& model_entity : inputs_[model_name]) {
					auto entity = (*states)[model_name].add_entities();
					for (auto& [output_name, value] : model_entity) {
						auto params = entity->mutable_params();
						(*params)[output_name] = AnyToMsg(value, data_[model_name][output_name]);
					}
				}
			}
			req.set_terminated(terminated);
			req.set_truncated(truncated);
			req.set_reward(0);
			streams_[agent_addr]->Write(req);
		}

		if (terminated || truncated) {
			grpc::ClientContext ctx;
			simenv::SimCmd req;
			types::CommonResponse res;
			req.set_type("episode");
			req.set_params("{}");
			simenv_->SimControl(&ctx, req, &res);

			state_ = CSInstanceState::IS_DESTROYED;
		}
		else {
			for (auto& [agent_addr, agent_models] : routes_) {
				types::SimAction res;
				if (streams_[agent_addr]->Read(&res)) {
					for (auto& [model_name, model_msg] : res.actions()) {
						auto entity = model_msg.entities(0);
						for (auto& [input_name, msg_value] : entity.params()) {
							outputs_[model_name][input_name] = MsgToAny(msg_value, data_[model_name][input_name]);
						}
					}
				}
			}

			state_ = CSInstanceState::IS_RUNNING;
		}
	}
	catch (std::exception& e) {
		WriteLog(e.what(), 5);
	}

	WriteLog("intel_proxy model Tick", 0);
	return true;
}

bool IntelProxy::SetInput(const std::unordered_map<std::string, std::any>& value) {
	if (sim_steps_ % sim_step_ratio_ != 0)
	{
		return true;
	}

	std::unordered_map<std::string, CSValueMap> tmp_inputs;
	for (auto& [param_name, value] : value)
	{
		auto idx = param_name.find("_output_");
		if (idx != std::string::npos) {
			auto model_name = param_name.substr(0, idx);
			auto output_name = param_name.substr(idx + 8);
			if (tmp_inputs.find(model_name) == tmp_inputs.end()) {
				tmp_inputs.emplace(model_name, CSValueMap());
			}
			tmp_inputs[model_name].emplace(output_name, value);
		}
	}
	for (auto& [model_name, model_entity] : tmp_inputs)
	{
		inputs_[model_name].emplace_back(model_entity);
	}

	WriteLog("intel_proxy model SetInput", 0);
	return true;
}

std::unordered_map<std::string, std::any>* IntelProxy::GetOutput() {
	if (sim_steps_++ % sim_step_ratio_ != 0)
	{
		return &params_;
	}

	params_.clear();
	params_.emplace("InstanceName", GetInstanceName());
	params_.emplace("ForceSideID", GetForceSideID());
	params_.emplace("ModelID", GetModelID());
	params_.emplace("ID", GetID());
	params_.emplace("State", uint16_t(GetState()));

	for (auto& [model_name, model_entity] : outputs_)
	{
		for (auto& [input_name, value] : model_entity)
		{
			auto param_name = model_name + "_input_" + input_name;
			params_.emplace(param_name, value);
		}
	}

	for (auto& it : inputs_)
	{
		it.second.clear();
	}

	WriteLog("intel_proxy model GetOutput", 0);
	return &params_;
}


IntelProxy::~IntelProxy() {
	for (auto& [agent_addr, agent_models] : routes_) {
		streams_[agent_addr]->WritesDone();
		streams_[agent_addr]->Finish();
	}

	if (hmodule) {
#ifdef _WIN32
		FreeLibrary(hmodule);
#else
		dlclose(hmodule);
#endif
	}
}

std::any IntelProxy::DefaultValue(const std::string& type) {
	std::any ret;
	auto idx = type.find("[]");
	if (idx == std::string::npos)
	{
		// 结构体类型
		if (types_.find(type) != types_.end()) {
			CSValueMap any_map;
			for (auto& [field_name, field_type] : types_[type]) {
				any_map.emplace(field_name, DefaultValue(field_type));
			}
			ret = any_map;
		}
		// 常用基本类型
		else if (type == "float64")
		{
			ret = (double)0;
		}
		else if (type == "uint64")
		{
			ret = (uint64_t)0;
		}
		else if (type == "uint16")
		{
			ret = (uint16_t)0;
		}
		else if (type == "bool")
		{
			ret = false;
		}
		else if (type == "int32")
		{
			ret = (int32_t)0;
		}
		else if (type == "string")
		{
			ret = std::string();
		}
		// 不常用基本类型
		else if (type == "float32")
		{
			ret = (float)0;
		}
		else if (type == "float128")
		{
			ret = (long double)0;
		}
		else if (type == "uint32")
		{
			ret = (uint32_t)0;
		}
		else if (type == "uin8")
		{
			ret = (uint8_t)0;
		}
		else if (type == "int64")
		{
			ret = (int64_t)0;
		}
		else if (type == "int16")
		{
			ret = (int16_t)0;
		}
		else if (type == "int8")
		{
			ret = (int8_t)0;
		}
		// 不支持的类型
		else
		{
			throw std::logic_error("Unsupported type!");
		}
	}
	else
	{
		// 数组类型
		CSValueVec any_vector;
		ret = any_vector;
	}
	return ret;
}

std::any IntelProxy::JsonToAny(const json& value, const std::string& type) {
	std::any ret;
	auto idx = type.find("[]");
	if (idx == std::string::npos)
	{
		// 结构体类型
		if (types_.find(type) != types_.end()) {
			CSValueMap any_map;
			for (auto& [field_name, field_type] : types_[type]) {
				any_map.emplace(field_name, JsonToAny(value[field_name], field_type));
			}
			ret = any_map;
		}
		// 常用基本类型
		else if (type == "float64")
		{
			ret = value.get<double>();
		}
		else if (type == "uint64")
		{
			ret = value.get<uint64_t>();
		}
		else if (type == "uint16")
		{
			ret = value.get<uint16_t>();
		}
		else if (type == "bool")
		{
			ret = value.get<bool>();
		}
		else if (type == "int32")
		{
			ret = value.get<int32_t>();
		}
		else if (type == "string")
		{
			ret = value.get<std::string>();
		}
		// 不常用基本类型
		else if (type == "float32")
		{
			ret = value.get<float>();
		}
		else if (type == "float128")
		{
			ret = value.get<long double>();
		}
		else if (type == "uint32")
		{
			ret = value.get<uint32_t>();
		}
		else if (type == "uin8")
		{
			ret = value.get<uint8_t>();
		}
		else if (type == "int64")
		{
			ret = value.get<int64_t>();
		}
		else if (type == "int16")
		{
			ret = value.get<int16_t>();
		}
		else if (type == "int8")
		{
			ret = value.get<int8_t>();
		}
		// 不支持的类型
		else
		{
			throw std::logic_error("Unsupported type!");
		}
	}
	else
	{
		// 数组类型
		CSValueVec any_vector;
		auto item_type = type.substr(0, idx);
		for (auto& it : value) {
			any_vector.emplace_back(JsonToAny(it, item_type));
		}
		ret = any_vector;
	}
	return ret;
}

types::SimParam IntelProxy::AnyToMsg(const std::any& value, const std::string& type) {
	types::SimParam ret;
	auto idx = type.find("[]");
	if (idx == std::string::npos)
	{
		// 结构体类型
		if (types_.find(type) != types_.end()) {
			auto struct_param = new types::SimParam_Struct();
			auto struct_fields = struct_param->mutable_fields();
			auto any_map = std::any_cast<CSValueMap>(value);
			for (auto& [field_name, field_type] : types_[type]) {
				(*struct_fields)[field_name] = AnyToMsg(any_map[field_name], field_type);
			}
			ret.set_allocated_vstruct(struct_param);
		}
		// 常用基本类型
		else if (type == "float64")
		{
			ret.set_vdouble(std::any_cast<double>(value));
		}
		else if (type == "uint64")
		{
			ret.set_vint32((int32_t)std::any_cast<uint64_t>(value));
		}
		else if (type == "uint16")
		{
			ret.set_vint32((int32_t)std::any_cast<uint16_t>(value));
		}
		else if (type == "bool")
		{
			ret.set_vbool(std::any_cast<bool>(value));
		}
		else if (type == "int32")
		{
			ret.set_vint32(std::any_cast<int32_t>(value));
		}
		else if (type == "string")
		{
			ret.set_vstring(std::any_cast<std::string>(value));
		}
		// 不常用基本类型
		else if (type == "float32")
		{
			ret.set_vdouble((double)std::any_cast<float>(value));
		}
		else if (type == "float128")
		{
			ret.set_vdouble((double)std::any_cast<long double>(value));
		}
		else if (type == "uint32")
		{
			ret.set_vint32((int32_t)std::any_cast<uint32_t>(value));
		}
		else if (type == "uint8")
		{
			ret.set_vint32((int32_t)std::any_cast<uint8_t>(value));
		}
		else if (type == "int64")
		{
			ret.set_vint32((int32_t)std::any_cast<uint64_t>(value));
		}
		else if (type == "int16")
		{
			ret.set_vint32((int32_t)std::any_cast<int16_t>(value));
		}
		else if (type == "int8")
		{
			ret.set_vint32((int32_t)std::any_cast<int8_t>(value));
		}
		// 不支持的类型
		else
		{
			throw std::logic_error("Unsupported type!");
		}
	}
	else
	{
		// 数组类型
		auto array_param = new types::SimParam_Array();
		auto any_vector = std::any_cast<CSValueVec>(value);
		auto item_type = type.substr(0, idx);
		for (auto& it : any_vector) {
		  auto array_item = array_param->add_items();
			*array_item = AnyToMsg(it, item_type);
		}
		ret.set_allocated_varray(array_param);
	}
	return ret;
}

std::any IntelProxy::MsgToAny(const types::SimParam& value, const std::string& type) {
	std::any ret;
	auto idx = type.find("[]");
	if (idx == std::string::npos)
	{
		// 结构体类型
		if (types_.find(type) != types_.end()) {
			auto struct_param = value.vstruct();
			auto struct_fields = struct_param.fields();
			CSValueMap any_map;
			for (auto& [field_name, field_type] : types_[type]) {
				any_map.emplace(field_name, MsgToAny(struct_fields[field_name], field_type));
			}
			ret = any_map;
		}
		// 常用基本类型
		else if (type == "float64")
		{
			ret = value.vdouble();
		}
		else if (type == "uint64")
		{
			ret = (uint64_t)value.vint32();
		}
		else if (type == "uint16")
		{
			ret = (uint16_t)value.vint32();
		}
		else if (type == "bool")
		{
			ret = value.vbool();
		}
		else if (type == "int32")
		{
			ret = value.vint32();
		}
		else if (type == "string")
		{
			ret = value.vstring();
		}
		// 不常用基本类型
		else if (type == "float32")
		{
			ret = (float)value.vdouble();
		}
		else if (type == "float128")
		{
			ret = (long double)value.vdouble();
		}
		else if (type == "uint32")
		{
			ret = (uint32_t)value.vint32();
		}
		else if (type == "uint8")
		{
			ret = (uint8_t)value.vint32();
		}
		else if (type == "int64")
		{
			ret = (int64_t)value.vint32();
		}
		else if (type == "int16")
		{
			ret = (int16_t)value.vint32();
		}
		else if (type == "int8")
		{
			ret = (int8_t)value.vint32();
		}
		// 不支持的类型
		else
		{
			throw std::logic_error("Unsupported type!");
		}
	}
	else
	{
		// 数组类型
		auto array_param = value.varray();
		auto array_items = array_param.items();
		CSValueVec any_vector;
		auto item_type = type.substr(0, idx);
		for (auto& it : array_items) {
			any_vector.emplace_back(MsgToAny(it, item_type));
		}
		ret = any_vector;
	}
	return ret;
}


INTEL_PROXY_EXPORT CSModelObject* CreateModelObject() {
	CSModelObject* model = new IntelProxy();
	return model;
}

INTEL_PROXY_EXPORT void DestroyMemory(void* mem, bool is_array) {
	if (is_array) {
		delete[]((IntelProxy*)mem);
	}
	else {
		delete ((IntelProxy*)mem);
	}
}
