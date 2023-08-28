// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: simenv.proto

#include "simenv.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace game {
namespace simenv {
constexpr SimenvConfig::SimenvConfig(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : name_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , args_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string){}
struct SimenvConfigDefaultTypeInternal {
  constexpr SimenvConfigDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~SimenvConfigDefaultTypeInternal() {}
  union {
    SimenvConfig _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT SimenvConfigDefaultTypeInternal _SimenvConfig_default_instance_;
constexpr SimCmd::SimCmd(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : type_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , params_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string){}
struct SimCmdDefaultTypeInternal {
  constexpr SimCmdDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~SimCmdDefaultTypeInternal() {}
  union {
    SimCmd _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT SimCmdDefaultTypeInternal _SimCmd_default_instance_;
constexpr SimInfo::SimInfo(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : state_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , data_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , logs_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string){}
struct SimInfoDefaultTypeInternal {
  constexpr SimInfoDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~SimInfoDefaultTypeInternal() {}
  union {
    SimInfo _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT SimInfoDefaultTypeInternal _SimInfo_default_instance_;
}  // namespace simenv
}  // namespace game
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_simenv_2eproto[3];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_simenv_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_simenv_2eproto = nullptr;

const uint32_t TableStruct_simenv_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::game::simenv::SimenvConfig, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::game::simenv::SimenvConfig, name_),
  PROTOBUF_FIELD_OFFSET(::game::simenv::SimenvConfig, args_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::game::simenv::SimCmd, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::game::simenv::SimCmd, type_),
  PROTOBUF_FIELD_OFFSET(::game::simenv::SimCmd, params_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::game::simenv::SimInfo, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::game::simenv::SimInfo, state_),
  PROTOBUF_FIELD_OFFSET(::game::simenv::SimInfo, data_),
  PROTOBUF_FIELD_OFFSET(::game::simenv::SimInfo, logs_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::game::simenv::SimenvConfig)},
  { 8, -1, -1, sizeof(::game::simenv::SimCmd)},
  { 16, -1, -1, sizeof(::game::simenv::SimInfo)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::game::simenv::_SimenvConfig_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::game::simenv::_SimCmd_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::game::simenv::_SimInfo_default_instance_),
};

const char descriptor_table_protodef_simenv_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\014simenv.proto\022\013game.simenv\032\013types.proto"
  "\"*\n\014SimenvConfig\022\014\n\004name\030\001 \001(\t\022\014\n\004args\030\002"
  " \001(\t\"&\n\006SimCmd\022\014\n\004type\030\001 \001(\t\022\016\n\006params\030\002"
  " \001(\t\"4\n\007SimInfo\022\r\n\005state\030\001 \001(\t\022\014\n\004data\030\002"
  " \001(\t\022\014\n\004logs\030\003 \001(\t2\331\003\n\006Simenv\022E\n\014ResetSe"
  "rvice\022\031.game.types.CommonRequest\032\032.game."
  "types.CommonResponse\022C\n\014QueryService\022\031.g"
  "ame.types.CommonRequest\032\030.game.types.Ser"
  "viceState\022G\n\017GetSimenvConfig\022\031.game.type"
  "s.CommonRequest\032\031.game.simenv.SimenvConf"
  "ig\022H\n\017SetSimenvConfig\022\031.game.simenv.Sime"
  "nvConfig\032\032.game.types.CommonResponse\022=\n\n"
  "SimControl\022\023.game.simenv.SimCmd\032\032.game.t"
  "ypes.CommonResponse\022=\n\nSimMonitor\022\031.game"
  ".types.CommonRequest\032\024.game.simenv.SimIn"
  "fo\0222\n\004Call\022\024.game.types.CallData\032\024.game."
  "types.CallDatab\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_simenv_2eproto_deps[1] = {
  &::descriptor_table_types_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_simenv_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_simenv_2eproto = {
  false, false, 662, descriptor_table_protodef_simenv_2eproto, "simenv.proto", 
  &descriptor_table_simenv_2eproto_once, descriptor_table_simenv_2eproto_deps, 1, 3,
  schemas, file_default_instances, TableStruct_simenv_2eproto::offsets,
  file_level_metadata_simenv_2eproto, file_level_enum_descriptors_simenv_2eproto, file_level_service_descriptors_simenv_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_simenv_2eproto_getter() {
  return &descriptor_table_simenv_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_simenv_2eproto(&descriptor_table_simenv_2eproto);
namespace game {
namespace simenv {

// ===================================================================

class SimenvConfig::_Internal {
 public:
};

SimenvConfig::SimenvConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:game.simenv.SimenvConfig)
}
SimenvConfig::SimenvConfig(const SimenvConfig& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_name().empty()) {
    name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_name(), 
      GetArenaForAllocation());
  }
  args_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    args_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_args().empty()) {
    args_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_args(), 
      GetArenaForAllocation());
  }
  // @@protoc_insertion_point(copy_constructor:game.simenv.SimenvConfig)
}

inline void SimenvConfig::SharedCtor() {
name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
args_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  args_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
}

SimenvConfig::~SimenvConfig() {
  // @@protoc_insertion_point(destructor:game.simenv.SimenvConfig)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void SimenvConfig::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  args_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void SimenvConfig::ArenaDtor(void* object) {
  SimenvConfig* _this = reinterpret_cast< SimenvConfig* >(object);
  (void)_this;
}
void SimenvConfig::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void SimenvConfig::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void SimenvConfig::Clear() {
// @@protoc_insertion_point(message_clear_start:game.simenv.SimenvConfig)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  name_.ClearToEmpty();
  args_.ClearToEmpty();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* SimenvConfig::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string name = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_name();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "game.simenv.SimenvConfig.name"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string args = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          auto str = _internal_mutable_args();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "game.simenv.SimenvConfig.args"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* SimenvConfig::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:game.simenv.SimenvConfig)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // string name = 1;
  if (!this->_internal_name().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_name().data(), static_cast<int>(this->_internal_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "game.simenv.SimenvConfig.name");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_name(), target);
  }

  // string args = 2;
  if (!this->_internal_args().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_args().data(), static_cast<int>(this->_internal_args().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "game.simenv.SimenvConfig.args");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_args(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:game.simenv.SimenvConfig)
  return target;
}

size_t SimenvConfig::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:game.simenv.SimenvConfig)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string name = 1;
  if (!this->_internal_name().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_name());
  }

  // string args = 2;
  if (!this->_internal_args().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_args());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData SimenvConfig::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    SimenvConfig::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*SimenvConfig::GetClassData() const { return &_class_data_; }

void SimenvConfig::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<SimenvConfig *>(to)->MergeFrom(
      static_cast<const SimenvConfig &>(from));
}


void SimenvConfig::MergeFrom(const SimenvConfig& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:game.simenv.SimenvConfig)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_name().empty()) {
    _internal_set_name(from._internal_name());
  }
  if (!from._internal_args().empty()) {
    _internal_set_args(from._internal_args());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void SimenvConfig::CopyFrom(const SimenvConfig& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:game.simenv.SimenvConfig)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SimenvConfig::IsInitialized() const {
  return true;
}

void SimenvConfig::InternalSwap(SimenvConfig* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &name_, lhs_arena,
      &other->name_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &args_, lhs_arena,
      &other->args_, rhs_arena
  );
}

::PROTOBUF_NAMESPACE_ID::Metadata SimenvConfig::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_simenv_2eproto_getter, &descriptor_table_simenv_2eproto_once,
      file_level_metadata_simenv_2eproto[0]);
}

// ===================================================================

class SimCmd::_Internal {
 public:
};

SimCmd::SimCmd(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:game.simenv.SimCmd)
}
SimCmd::SimCmd(const SimCmd& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  type_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    type_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_type().empty()) {
    type_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_type(), 
      GetArenaForAllocation());
  }
  params_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    params_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_params().empty()) {
    params_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_params(), 
      GetArenaForAllocation());
  }
  // @@protoc_insertion_point(copy_constructor:game.simenv.SimCmd)
}

inline void SimCmd::SharedCtor() {
type_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  type_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
params_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  params_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
}

SimCmd::~SimCmd() {
  // @@protoc_insertion_point(destructor:game.simenv.SimCmd)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void SimCmd::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  type_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  params_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void SimCmd::ArenaDtor(void* object) {
  SimCmd* _this = reinterpret_cast< SimCmd* >(object);
  (void)_this;
}
void SimCmd::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void SimCmd::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void SimCmd::Clear() {
// @@protoc_insertion_point(message_clear_start:game.simenv.SimCmd)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  type_.ClearToEmpty();
  params_.ClearToEmpty();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* SimCmd::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string type = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_type();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "game.simenv.SimCmd.type"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string params = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          auto str = _internal_mutable_params();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "game.simenv.SimCmd.params"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* SimCmd::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:game.simenv.SimCmd)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // string type = 1;
  if (!this->_internal_type().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_type().data(), static_cast<int>(this->_internal_type().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "game.simenv.SimCmd.type");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_type(), target);
  }

  // string params = 2;
  if (!this->_internal_params().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_params().data(), static_cast<int>(this->_internal_params().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "game.simenv.SimCmd.params");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_params(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:game.simenv.SimCmd)
  return target;
}

size_t SimCmd::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:game.simenv.SimCmd)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string type = 1;
  if (!this->_internal_type().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_type());
  }

  // string params = 2;
  if (!this->_internal_params().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_params());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData SimCmd::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    SimCmd::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*SimCmd::GetClassData() const { return &_class_data_; }

void SimCmd::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<SimCmd *>(to)->MergeFrom(
      static_cast<const SimCmd &>(from));
}


void SimCmd::MergeFrom(const SimCmd& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:game.simenv.SimCmd)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_type().empty()) {
    _internal_set_type(from._internal_type());
  }
  if (!from._internal_params().empty()) {
    _internal_set_params(from._internal_params());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void SimCmd::CopyFrom(const SimCmd& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:game.simenv.SimCmd)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SimCmd::IsInitialized() const {
  return true;
}

void SimCmd::InternalSwap(SimCmd* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &type_, lhs_arena,
      &other->type_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &params_, lhs_arena,
      &other->params_, rhs_arena
  );
}

::PROTOBUF_NAMESPACE_ID::Metadata SimCmd::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_simenv_2eproto_getter, &descriptor_table_simenv_2eproto_once,
      file_level_metadata_simenv_2eproto[1]);
}

// ===================================================================

class SimInfo::_Internal {
 public:
};

SimInfo::SimInfo(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:game.simenv.SimInfo)
}
SimInfo::SimInfo(const SimInfo& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  state_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    state_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_state().empty()) {
    state_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_state(), 
      GetArenaForAllocation());
  }
  data_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    data_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_data().empty()) {
    data_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_data(), 
      GetArenaForAllocation());
  }
  logs_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    logs_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_logs().empty()) {
    logs_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_logs(), 
      GetArenaForAllocation());
  }
  // @@protoc_insertion_point(copy_constructor:game.simenv.SimInfo)
}

inline void SimInfo::SharedCtor() {
state_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  state_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
data_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  data_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
logs_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  logs_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), "", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
}

SimInfo::~SimInfo() {
  // @@protoc_insertion_point(destructor:game.simenv.SimInfo)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void SimInfo::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  state_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  data_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  logs_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void SimInfo::ArenaDtor(void* object) {
  SimInfo* _this = reinterpret_cast< SimInfo* >(object);
  (void)_this;
}
void SimInfo::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void SimInfo::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void SimInfo::Clear() {
// @@protoc_insertion_point(message_clear_start:game.simenv.SimInfo)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  state_.ClearToEmpty();
  data_.ClearToEmpty();
  logs_.ClearToEmpty();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* SimInfo::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string state = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_state();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "game.simenv.SimInfo.state"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string data = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          auto str = _internal_mutable_data();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "game.simenv.SimInfo.data"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string logs = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 26)) {
          auto str = _internal_mutable_logs();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "game.simenv.SimInfo.logs"));
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* SimInfo::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:game.simenv.SimInfo)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // string state = 1;
  if (!this->_internal_state().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_state().data(), static_cast<int>(this->_internal_state().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "game.simenv.SimInfo.state");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_state(), target);
  }

  // string data = 2;
  if (!this->_internal_data().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_data().data(), static_cast<int>(this->_internal_data().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "game.simenv.SimInfo.data");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_data(), target);
  }

  // string logs = 3;
  if (!this->_internal_logs().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_logs().data(), static_cast<int>(this->_internal_logs().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "game.simenv.SimInfo.logs");
    target = stream->WriteStringMaybeAliased(
        3, this->_internal_logs(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:game.simenv.SimInfo)
  return target;
}

size_t SimInfo::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:game.simenv.SimInfo)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string state = 1;
  if (!this->_internal_state().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_state());
  }

  // string data = 2;
  if (!this->_internal_data().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_data());
  }

  // string logs = 3;
  if (!this->_internal_logs().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_logs());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData SimInfo::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    SimInfo::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*SimInfo::GetClassData() const { return &_class_data_; }

void SimInfo::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<SimInfo *>(to)->MergeFrom(
      static_cast<const SimInfo &>(from));
}


void SimInfo::MergeFrom(const SimInfo& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:game.simenv.SimInfo)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_state().empty()) {
    _internal_set_state(from._internal_state());
  }
  if (!from._internal_data().empty()) {
    _internal_set_data(from._internal_data());
  }
  if (!from._internal_logs().empty()) {
    _internal_set_logs(from._internal_logs());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void SimInfo::CopyFrom(const SimInfo& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:game.simenv.SimInfo)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SimInfo::IsInitialized() const {
  return true;
}

void SimInfo::InternalSwap(SimInfo* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &state_, lhs_arena,
      &other->state_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &data_, lhs_arena,
      &other->data_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &logs_, lhs_arena,
      &other->logs_, rhs_arena
  );
}

::PROTOBUF_NAMESPACE_ID::Metadata SimInfo::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_simenv_2eproto_getter, &descriptor_table_simenv_2eproto_once,
      file_level_metadata_simenv_2eproto[2]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace simenv
}  // namespace game
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::game::simenv::SimenvConfig* Arena::CreateMaybeMessage< ::game::simenv::SimenvConfig >(Arena* arena) {
  return Arena::CreateMessageInternal< ::game::simenv::SimenvConfig >(arena);
}
template<> PROTOBUF_NOINLINE ::game::simenv::SimCmd* Arena::CreateMaybeMessage< ::game::simenv::SimCmd >(Arena* arena) {
  return Arena::CreateMessageInternal< ::game::simenv::SimCmd >(arena);
}
template<> PROTOBUF_NOINLINE ::game::simenv::SimInfo* Arena::CreateMaybeMessage< ::game::simenv::SimInfo >(Arena* arena) {
  return Arena::CreateMessageInternal< ::game::simenv::SimInfo >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
