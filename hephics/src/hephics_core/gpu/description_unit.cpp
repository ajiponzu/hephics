
#include "../gpu.hpp"

hpxc::gpu::DescriptionUnit::DescriptionUnit(
    const std::unordered_map<std::string, ShaderModule>& shader_module_map,
    const std::vector<std::string>& module_keys) {
  for (const auto& module_key : module_keys) {
    const auto& shader_module = shader_module_map.at(module_key);

    for (const auto& [key, descriptor_info] :
         shader_module.getDescriptorInfoMap()) {
      if (m_descriptorInfoMap.contains(key)) {
        // merge stage flags because the same descriptor info is used in
        m_descriptorInfoMap[key].stage_flags |= descriptor_info.stage_flags;
        continue;
      }
      m_descriptorInfoMap[key] = descriptor_info;
    }
    for (const auto& [key, push_constant_range] :
         shader_module.getPushConstantRangeMap()) {
      if (m_pushConstantRangeMap.contains(key)) {
        // merge stage flags because the same push constants is used in
        m_pushConstantRangeMap[key].stage_flags |=
            push_constant_range.stage_flags;
        continue;
      }
      m_pushConstantRangeMap[key] = push_constant_range;
    }
  }
}

hpxc::gpu::DescriptionUnit::~DescriptionUnit() {}
