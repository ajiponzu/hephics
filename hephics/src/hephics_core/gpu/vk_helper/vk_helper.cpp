#include "../vk_helper.hpp"

vk::MemoryPropertyFlags vk_helper::getMemoryPropertyFlags(
    const hpxc::MemoryUsage memory_usage) {
  using MemoryUsage = hpxc::MemoryUsage;

  switch (memory_usage) {
    case MemoryUsage::GpuOnly:
      return vk::MemoryPropertyFlagBits::eDeviceLocal;
    case MemoryUsage::CpuOnly:
      return vk::MemoryPropertyFlagBits::eHostVisible |
             vk::MemoryPropertyFlagBits::eHostCoherent;
    case MemoryUsage::CpuToGpu:
      return vk::MemoryPropertyFlagBits::eHostVisible |
             vk::MemoryPropertyFlagBits::eDeviceLocal;
    case MemoryUsage::GpuToCpu:
      return vk::MemoryPropertyFlagBits::eHostVisible |
             vk::MemoryPropertyFlagBits::eHostCached;
    default:
      return vk::MemoryPropertyFlagBits::eDeviceLocal;
  }
}

vk::AccessFlagBits vk_helper::getAccessFlagBits(
    const hpxc::AccessFlag access_flag) {
  using AccessFlag = hpxc::AccessFlag;

  switch (access_flag) {
    case AccessFlag::IndirectCommandRead:
      return vk::AccessFlagBits::eIndirectCommandRead;
    case AccessFlag::IndexRead:
      return vk::AccessFlagBits::eIndexRead;
    case AccessFlag::VertexAttributeRead:
      return vk::AccessFlagBits::eVertexAttributeRead;
    case AccessFlag::UniformRead:
      return vk::AccessFlagBits::eUniformRead;
    case AccessFlag::InputAttachmentRead:
      return vk::AccessFlagBits::eInputAttachmentRead;
    case AccessFlag::ShaderRead:
      return vk::AccessFlagBits::eShaderRead;
    case AccessFlag::ShaderWrite:
      return vk::AccessFlagBits::eShaderWrite;
    case AccessFlag::ColorAttachmentRead:
      return vk::AccessFlagBits::eColorAttachmentRead;
    case AccessFlag::ColorAttachmentWrite:
      return vk::AccessFlagBits::eColorAttachmentWrite;
    case AccessFlag::DepthStencilAttachmentRead:
      return vk::AccessFlagBits::eDepthStencilAttachmentRead;
    case AccessFlag::DepthStencilAttachmentWrite:
      return vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    case AccessFlag::TransferRead:
      return vk::AccessFlagBits::eTransferRead;
    case AccessFlag::TransferWrite:
      return vk::AccessFlagBits::eTransferWrite;
    case AccessFlag::HostRead:
      return vk::AccessFlagBits::eHostRead;
    case AccessFlag::HostWrite:
      return vk::AccessFlagBits::eHostWrite;
    case AccessFlag::MemoryRead:
      return vk::AccessFlagBits::eMemoryRead;
    case AccessFlag::MemoryWrite:
      return vk::AccessFlagBits::eMemoryWrite;
    default:
      return vk::AccessFlagBits::eNone;
  }
}

vk::PipelineStageFlagBits vk_helper::getPipelineStageFlagBits(
    const hpxc::PipelineStage stage) {
  using PipelineStage = hpxc::PipelineStage;

  switch (stage) {
    case PipelineStage::TopOfPipe:
      return vk::PipelineStageFlagBits::eTopOfPipe;
    case PipelineStage::DrawIndirect:
      return vk::PipelineStageFlagBits::eDrawIndirect;
    case PipelineStage::VertexInput:
      return vk::PipelineStageFlagBits::eVertexInput;
    case PipelineStage::VertexShader:
      return vk::PipelineStageFlagBits::eVertexShader;
    case PipelineStage::TessellationControlShader:
      return vk::PipelineStageFlagBits::eTessellationControlShader;
    case PipelineStage::TessellationEvaluationShader:
      return vk::PipelineStageFlagBits::eTessellationEvaluationShader;
    case PipelineStage::GeometryShader:
      return vk::PipelineStageFlagBits::eGeometryShader;
    case PipelineStage::FragmentShader:
      return vk::PipelineStageFlagBits::eFragmentShader;
    case PipelineStage::EarlyFragmentTests:
      return vk::PipelineStageFlagBits::eEarlyFragmentTests;
    case PipelineStage::LateFragmentTests:
      return vk::PipelineStageFlagBits::eLateFragmentTests;
    case PipelineStage::ColorAttachmentOutput:
      return vk::PipelineStageFlagBits::eColorAttachmentOutput;
    case PipelineStage::ComputeShader:
      return vk::PipelineStageFlagBits::eComputeShader;
    case PipelineStage::Transfer:
      return vk::PipelineStageFlagBits::eTransfer;
    case PipelineStage::BottomOfPipe:
      return vk::PipelineStageFlagBits::eBottomOfPipe;
    case PipelineStage::Host:
      return vk::PipelineStageFlagBits::eHost;
    case PipelineStage::AllGraphics:
      return vk::PipelineStageFlagBits::eAllGraphics;
    case PipelineStage::AllCommands:
      return vk::PipelineStageFlagBits::eAllCommands;
    default:
      return vk::PipelineStageFlagBits::eTopOfPipe;
  }
}

vk::ImageLayout vk_helper::getImageLayout(
    const hpxc::ImageLayout image_layout) {
  using ImageLayout = hpxc::ImageLayout;

  switch (image_layout) {
    case ImageLayout::Undefined:
      return vk::ImageLayout::eUndefined;
    case ImageLayout::General:
      return vk::ImageLayout::eGeneral;
    case ImageLayout::ColorAttachmentOptimal:
      return vk::ImageLayout::eColorAttachmentOptimal;
    case ImageLayout::DepthStencilAttachmentOptimal:
      return vk::ImageLayout::eDepthStencilAttachmentOptimal;
    case ImageLayout::DepthStencilReadOnlyOptimal:
      return vk::ImageLayout::eDepthStencilReadOnlyOptimal;
    case ImageLayout::ShaderReadOnlyOptimal:
      return vk::ImageLayout::eShaderReadOnlyOptimal;
    case ImageLayout::TransferSrcOptimal:
      return vk::ImageLayout::eTransferSrcOptimal;
    case ImageLayout::TransferDstOptimal:
      return vk::ImageLayout::eTransferDstOptimal;
    case ImageLayout::Preinitialized:
      return vk::ImageLayout::ePreinitialized;
    case ImageLayout::PresentSrc:
      return vk::ImageLayout::ePresentSrcKHR;
    case ImageLayout::SharedPresent:
      return vk::ImageLayout::eSharedPresentKHR;
    case ImageLayout::DepthReadOnlyStencilAttachmentOptimal:
      return vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal;
    case ImageLayout::DepthAttachmentStencilReadOnlyOptimal:
      return vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal;
    default:
      return vk::ImageLayout::eUndefined;
  }
}

vk::Format vk_helper::getImageFormat(const hpxc::ImageFormat image_format) {
  using ImageFormat = hpxc::ImageFormat;

  switch (image_format) {
    case ImageFormat::R8Unorm:
      return vk::Format::eR8Unorm;
    case ImageFormat::R8Snorm:
      return vk::Format::eR8Snorm;
    case ImageFormat::R8Uscaled:
      return vk::Format::eR8Uscaled;
    case ImageFormat::R8Sscaled:
      return vk::Format::eR8Sscaled;
    case ImageFormat::R8Uint:
      return vk::Format::eR8Uint;
    case ImageFormat::R8Sint:
      return vk::Format::eR8Sint;
    case ImageFormat::R8Srgb:
      return vk::Format::eR8Srgb;
    case ImageFormat::R8G8Unorm:
      return vk::Format::eR8G8Unorm;
    case ImageFormat::R8G8Snorm:
      return vk::Format::eR8G8Snorm;
    case ImageFormat::R8G8Uscaled:
      return vk::Format::eR8G8Uscaled;
    case ImageFormat::R8G8Sscaled:
      return vk::Format::eR8G8Sscaled;
    case ImageFormat::R8G8Uint:
      return vk::Format::eR8G8Uint;
    case ImageFormat::R8G8Sint:
      return vk::Format::eR8G8Sint;
    case ImageFormat::R8G8Srgb:
      return vk::Format::eR8G8Srgb;
    case ImageFormat::R8G8B8Unorm:
      return vk::Format::eR8G8B8Unorm;
    case ImageFormat::R8G8B8Snorm:
      return vk::Format::eR8G8B8Snorm;
    case ImageFormat::R8G8B8Uscaled:
      return vk::Format::eR8G8B8Uscaled;
    case ImageFormat::R8G8B8Sscaled:
      return vk::Format::eR8G8B8Sscaled;
    case ImageFormat::R8G8B8Uint:
      return vk::Format::eR8G8B8Uint;
    case ImageFormat::R8G8B8Sint:
      return vk::Format::eR8G8B8Sint;
    case ImageFormat::R8G8B8Srgb:
      return vk::Format::eR8G8B8Srgb;
    case ImageFormat::R8G8B8A8Unorm:
      return vk::Format::eR8G8B8A8Unorm;
    case ImageFormat::R8G8B8A8Snorm:
      return vk::Format::eR8G8B8A8Snorm;
    case ImageFormat::R8G8B8A8Uscaled:
      return vk::Format::eR8G8B8A8Uscaled;
    case ImageFormat::R8G8B8A8Sscaled:
      return vk::Format::eR8G8B8A8Sscaled;
    case ImageFormat::R8G8B8A8Uint:
      return vk::Format::eR8G8B8A8Uint;
    case ImageFormat::R8G8B8A8Sint:
      return vk::Format::eR8G8B8A8Sint;
    case ImageFormat::R8G8B8A8Srgb:
      return vk::Format::eR8G8B8A8Srgb;
    default:
      return vk::Format::eR8G8B8A8Unorm;
  }
}

vk::ImageAspectFlags vk_helper::getImageAspectFlags(
    const hpxc::ImageAspect image_aspect) {
  using ImageAspect = hpxc::ImageAspect;

  vk::ImageSubresourceRange subresource_range;
  switch (image_aspect) {
    case ImageAspect::Color:
      return vk::ImageAspectFlagBits::eColor;
    case ImageAspect::Depth:
      return vk::ImageAspectFlagBits::eDepth;
    case ImageAspect::Stencil:
      return vk::ImageAspectFlagBits::eStencil;
    case ImageAspect::DepthStencil:
      return vk::ImageAspectFlagBits::eDepth |
             vk::ImageAspectFlagBits::eStencil;
    default:
      return vk::ImageAspectFlagBits::eColor;
  }
}

vk::ShaderStageFlags vk_helper::getShaderStageFlagBits(
    const hpxc::ShaderStage shader_stage) {
  using ShaderStage = hpxc::ShaderStage;

  switch (shader_stage) {
    case ShaderStage::Vertex:
      return vk::ShaderStageFlagBits::eVertex;
    case ShaderStage::Fragment:
      return vk::ShaderStageFlagBits::eFragment;
    case ShaderStage::Compute:
      return vk::ShaderStageFlagBits::eCompute;
    default:
      return vk::ShaderStageFlagBits::eVertex;
  }
}
