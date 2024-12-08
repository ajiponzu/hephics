#include "../gpu.hpp"

hpxc::gpu::Pipeline::Pipeline(
    const std::unique_ptr<Context>& ptr_context,
    const DescriptionUnit& description_unit,
    const DescriptorSetLayout& descriptor_set_layout) {
  std::vector<vk::PushConstantRange> push_constant_ranges;
  for (const auto& [_, push_constant_range] :
       description_unit.getPushConstantRangeMap()) {
    push_constant_ranges.emplace_back(
        push_constant_range.stage_flags, push_constant_range.offset,
        static_cast<uint32_t>(push_constant_range.size));
  }

  vk::PipelineLayoutCreateInfo pipeline_layout_info;
  pipeline_layout_info.setSetLayouts(
      descriptor_set_layout.getDescriptorSetLayout().get());
  pipeline_layout_info.setPushConstantRanges(push_constant_ranges);

  m_ptrPipelineLayout =
      ptr_context->getDevice()->getLogicalDevice()->createPipelineLayoutUnique(
          pipeline_layout_info);
}

hpxc::gpu::Pipeline::~Pipeline() {}

void hpxc::gpu::Pipeline::constructComputePipeline(
    const std::unique_ptr<Context>& ptr_context,
    const ShaderModule& shader_module) {
  m_queueFamilyType = QueueFamilyType::Compute;

  vk::PipelineShaderStageCreateInfo shader_stage_info;
  shader_stage_info.setStage(vk::ShaderStageFlagBits::eCompute);
  shader_stage_info.setModule(shader_module.getModule().get());
  shader_stage_info.setPName(shader_module.getEntryPointName().c_str());

  vk::ComputePipelineCreateInfo compute_pipeline_info;
  compute_pipeline_info.setLayout(m_ptrPipelineLayout.get());
  compute_pipeline_info.setStage(shader_stage_info);

  m_ptrPipeline =
      ptr_context->getDevice()
          ->getLogicalDevice()
          ->createComputePipelineUnique(nullptr, compute_pipeline_info)
          .value;
}
