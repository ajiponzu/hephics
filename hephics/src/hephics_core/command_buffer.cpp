#include <iostream>

#include "../hephics_core.hpp"
#include "gpu/vk_helper.hpp"

void hpxc::CommandBuffer::begin(
    const CommandBeginInfo& command_begin_info) const {
  vk::CommandBufferInheritanceInfo inheritance_info;
  inheritance_info.setRenderPass(nullptr);
  inheritance_info.setSubpass(command_begin_info.subpass_index);
  inheritance_info.setFramebuffer(nullptr);
  inheritance_info.setQueryFlags(vk::QueryControlFlags(0U));
  inheritance_info.setPipelineStatistics(vk::QueryPipelineStatisticFlags(0U));
  inheritance_info.setOcclusionQueryEnable(VK_FALSE);

  vk::CommandBufferBeginInfo begin_info;
  begin_info.setPInheritanceInfo(&inheritance_info);
  begin_info.setFlags(command_begin_info.usage_flags);

  m_commandBuffer.begin(begin_info);
}

void hpxc::CommandBuffer::end() const { m_commandBuffer.end(); }

void hpxc::CommandBuffer::setPipelineBarrier(
    const gpu::BufferBarrier& barrier, const PipelineStage src_stage,
    const PipelineStage dst_stage) const {
  m_commandBuffer.pipelineBarrier(
      vk_helper::getPipelineStageFlagBits(src_stage),
      vk_helper::getPipelineStageFlagBits(dst_stage),
      vk::DependencyFlagBits(0U), nullptr, barrier.getBarrier(), nullptr);
}

void hpxc::CommandBuffer::setPipelineBarrier(
    const gpu::ImageBarrier& barrier, const PipelineStage src_stage,
    const PipelineStage dst_stage) const {
  m_commandBuffer.pipelineBarrier(
      vk_helper::getPipelineStageFlagBits(src_stage),
      vk_helper::getPipelineStageFlagBits(dst_stage),
      vk::DependencyFlagBits(0U), nullptr, nullptr, barrier.getBarrier());
}

void hpxc::CommandBuffer::pushConstants(
    const gpu::Pipeline& pipeline, const std::vector<ShaderStage>& dst_stages,
    const uint32_t offset, const std::vector<float_t>& data) const {
  vk::ShaderStageFlags vk_stages;
  for (const auto& stage : dst_stages) {
    vk_stages |= vk_helper::getShaderStageFlagBits(stage);
  }

  m_commandBuffer.pushConstants(
      pipeline.getPipelineLayout().get(), vk_stages, offset,
      static_cast<uint32_t>(sizeof(float_t) * data.size()), data.data());
}

void hpxc::CommandBuffer::resetCommands() const {
  m_commandBuffer.reset(vk::CommandBufferResetFlags());
}

void hpxc::TransferCommandBuffer::copyBuffer(
    const gpu::Buffer& staging_buffer, const gpu::Buffer& dst_buffer) const {
  vk::BufferCopy copy_region;
  copy_region.setSize(staging_buffer.getSize());

  m_commandBuffer.copyBuffer(staging_buffer.getBuffer(), dst_buffer.getBuffer(),
                             copy_region);
}

void hpxc::TransferCommandBuffer::copyBufferToImage(
    const gpu::Buffer& buffer, const gpu::Image& image,
    const ImageLayout image_layout,
    const ImageViewInfo& image_view_info) const {
  vk::BufferImageCopy copy_region;

  {
    vk::ImageSubresourceLayers subresource;

    subresource.setAspectMask(
        vk_helper::getImageAspectFlags(image_view_info.aspect));
    subresource.setMipLevel(image_view_info.base_mip_level);
    subresource.setBaseArrayLayer(image_view_info.base_array_layer);
    subresource.setLayerCount(image_view_info.array_layers);

    copy_region.setImageSubresource(subresource);
  }

  copy_region.setImageOffset({0U, 0U, 0U});

  const auto& graphical_size = image.getGraphicalSize();
  copy_region.setBufferRowLength(graphical_size.width);
  copy_region.setBufferImageHeight(graphical_size.height);
  copy_region.setImageExtent(
      {graphical_size.width, graphical_size.height, graphical_size.depth});

  vk::ImageLayout vk_image_layout = vk_helper::getImageLayout(image_layout);
  if (vk_image_layout != vk::ImageLayout::eTransferDstOptimal &&
      vk_image_layout != vk::ImageLayout::eGeneral &&
      vk_image_layout != vk::ImageLayout::eSharedPresentKHR) {
    vk_image_layout = vk::ImageLayout::eTransferDstOptimal;
  }

  m_commandBuffer.copyBufferToImage(buffer.getBuffer(), image.getImage(),
                                    vk_image_layout, copy_region);
}

void hpxc::TransferCommandBuffer::copyImageToBuffer(
    const gpu::Image& image, const gpu::Buffer& buffer,
    const ImageLayout image_layout,
    const ImageViewInfo& image_view_info) const {
  vk::BufferImageCopy copy_region;

  {
    vk::ImageSubresourceLayers subresource;

    subresource.setAspectMask(
        vk_helper::getImageAspectFlags(image_view_info.aspect));
    subresource.setMipLevel(image_view_info.base_mip_level);
    subresource.setBaseArrayLayer(image_view_info.base_array_layer);
    subresource.setLayerCount(image_view_info.array_layers);

    copy_region.setImageSubresource(subresource);
  }

  copy_region.setImageOffset({0U, 0U, 0U});

  const auto& graphical_size = image.getGraphicalSize();
  copy_region.setBufferRowLength(graphical_size.width);
  copy_region.setBufferImageHeight(graphical_size.height);
  copy_region.setImageExtent(
      {graphical_size.width, graphical_size.height, graphical_size.depth});

  vk::ImageLayout vk_image_layout = vk_helper::getImageLayout(image_layout);
  if (vk_image_layout != vk::ImageLayout::eTransferSrcOptimal &&
      vk_image_layout != vk::ImageLayout::eGeneral &&
      vk_image_layout != vk::ImageLayout::eSharedPresentKHR) {
    vk_image_layout = vk::ImageLayout::eTransferSrcOptimal;
  }

  m_commandBuffer.copyImageToBuffer(image.getImage(), vk_image_layout,
                                    buffer.getBuffer(), copy_region);
}

void hpxc::TransferCommandBuffer::setMipmaps(
    const gpu::Image& image, const PipelineStage dst_stage) const {
  ImageViewInfo image_view_info{};
  image_view_info.aspect = hpxc::ImageAspect::Color;
  image_view_info.base_mip_level = 0U;
  image_view_info.mip_levels = 1U;  // for sending each mip level
  image_view_info.base_array_layer = 0U;
  image_view_info.array_layers = 1U;

  const gpu::ImageBarrier src_image_barrier(
      image, {AccessFlag::TransferWrite}, {AccessFlag::TransferRead},
      ImageLayout::TransferDstOptimal, ImageLayout::TransferSrcOptimal,
      image_view_info);

  const gpu::ImageBarrier dst_image_barrier(
      image, {AccessFlag::TransferRead}, {AccessFlag::ShaderRead},
      ImageLayout::TransferSrcOptimal, ImageLayout::ShaderReadOnlyOptimal,
      image_view_info);

  vk::ImageMemoryBarrier src_barrier = src_image_barrier.getBarrier();
  vk::ImageMemoryBarrier dst_barrier = dst_image_barrier.getBarrier();

  if (dst_stage == PipelineStage::Transfer) {
    dst_barrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
    dst_barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
  }
  if (dst_stage == PipelineStage::BottomOfPipe) {
    dst_barrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
  }

  uint32_t mip_width = image.getGraphicalSize().width;
  uint32_t mip_height = image.getGraphicalSize().height;

  uint32_t mip_level = 1U;
  for (; mip_level < image_view_info.mip_levels; mip_level += 1U) {
    src_barrier.subresourceRange.setBaseMipLevel(mip_level - 1U);
    m_commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eTransfer,
                                    vk::DependencyFlagBits(0U), nullptr,
                                    nullptr, src_barrier);

    vk::ImageBlit blit;
    {
      vk::ImageSubresourceLayers subresource;

      subresource.setAspectMask(vk::ImageAspectFlagBits::eColor);
      subresource.setMipLevel(mip_level - 1U);
      subresource.setBaseArrayLayer(0U);
      subresource.setLayerCount(1U);
      blit.setSrcSubresource(subresource);
      blit.setSrcOffsets({vk::Offset3D(0U, 0U, 0U),
                          vk::Offset3D(static_cast<int32_t>(mip_width),
                                       static_cast<int32_t>(mip_height), 1U)});
    }
    {
      vk::ImageSubresourceLayers subresource;

      subresource.setAspectMask(vk::ImageAspectFlagBits::eColor);
      subresource.setMipLevel(mip_level);
      subresource.setBaseArrayLayer(0);
      subresource.setLayerCount(1);
      blit.setDstSubresource(subresource);
      blit.setDstOffsets(
          {vk::Offset3D(0U, 0U, 0U),
           vk::Offset3D(static_cast<int32_t>(std::max(1U, mip_width / 2U)),
                        static_cast<int32_t>(std::max(1U, mip_height / 2U)),
                        1U)});
    }

    m_commandBuffer.blitImage(
        image.getImage(), vk::ImageLayout::eTransferSrcOptimal,
        image.getImage(), vk::ImageLayout::eTransferDstOptimal, blit,
        vk::Filter::eLinear);

    dst_barrier.subresourceRange.setBaseMipLevel(mip_level - 1U);

    m_commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk_helper::getPipelineStageFlagBits(dst_stage),
        vk::DependencyFlagBits(0U), nullptr, nullptr, dst_barrier);

    mip_width = std::max(1U, mip_width / 2U);
    mip_height = std::max(1U, mip_height / 2U);
  }

  dst_barrier.subresourceRange.setBaseMipLevel(mip_level - 1);
  dst_barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);

  m_commandBuffer.pipelineBarrier(
      vk::PipelineStageFlagBits::eTransfer,
      vk_helper::getPipelineStageFlagBits(dst_stage), vk::DependencyFlagBits(0),
      nullptr, nullptr, dst_barrier);
}

void hpxc::TransferCommandBuffer::transferMipmapImages(
    const gpu::Image& image, const PipelineStage src_stage,
    const PipelineStage dst_stage,
    std::pair<uint32_t, uint32_t> queue_family_index) const {
  ImageViewInfo image_view_info{};
  image_view_info.aspect = hpxc::ImageAspect::Color;
  image_view_info.base_mip_level = 0U;
  image_view_info.mip_levels = 1U;
  image_view_info.base_array_layer = 0U;
  image_view_info.array_layers = 1U;

  const gpu::ImageBarrier image_barrier(
      image, {AccessFlag::TransferWrite}, {AccessFlag::ShaderRead},
      ImageLayout::TransferDstOptimal, ImageLayout::TransferDstOptimal,
      image_view_info);

  vk::ImageMemoryBarrier barrier = image_barrier.getBarrier();
  barrier.setSrcQueueFamilyIndex(queue_family_index.first);
  barrier.setDstQueueFamilyIndex(queue_family_index.second);

  uint32_t mip_level = 1U;
  for (; mip_level <= image.getMipLevels(); mip_level += 1U) {
    barrier.subresourceRange.setBaseMipLevel(mip_level - 1U);

    m_commandBuffer.pipelineBarrier(
        vk_helper::getPipelineStageFlagBits(src_stage),
        vk_helper::getPipelineStageFlagBits(dst_stage),
        vk::DependencyFlagBits(0U), nullptr, nullptr, barrier);
  }
}

void hpxc::TransferCommandBuffer::acquireMipmapImages(
    const gpu::Image& image, const PipelineStage src_stage,
    const PipelineStage dst_stage,
    std::pair<uint32_t, uint32_t> queue_family_index) const {
  ImageViewInfo image_view_info{};
  image_view_info.aspect = hpxc::ImageAspect::Color;
  image_view_info.base_mip_level = 0U;
  image_view_info.mip_levels = 1U;
  image_view_info.base_array_layer = 0U;
  image_view_info.array_layers = 1U;

  const gpu::ImageBarrier image_barrier(
      image, {AccessFlag::TransferWrite}, {AccessFlag::ShaderRead},
      ImageLayout::TransferDstOptimal, ImageLayout::ShaderReadOnlyOptimal,
      image_view_info);

  vk::ImageMemoryBarrier barrier = image_barrier.getBarrier();
  barrier.setSrcQueueFamilyIndex(queue_family_index.first);
  barrier.setDstQueueFamilyIndex(queue_family_index.second);

  uint32_t mip_level = 1U;
  for (; mip_level <= image.getMipLevels(); mip_level += 1U) {
    barrier.subresourceRange.setBaseMipLevel(mip_level - 1U);

    m_commandBuffer.pipelineBarrier(
        vk_helper::getPipelineStageFlagBits(src_stage),
        vk_helper::getPipelineStageFlagBits(dst_stage),
        vk::DependencyFlagBits(0U), nullptr, nullptr, barrier);
  }
}

void hpxc::ComputeCommandBuffer::compute(
    const gpu::Pipeline& pipeline, const gpu::DescriptorSet& descriptor_set,
    const ComputeWorkGroupSize& work_group_size) const {
  if (pipeline.getQueueFamilyType() != QueueFamilyType::Compute) {
    std::cerr << "argument pipeline is not compute pipeline." << std::endl;

    return;
  }

  m_commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                               pipeline.getPipeline().get());
  m_commandBuffer.bindDescriptorSets(
      vk::PipelineBindPoint::eCompute, pipeline.getPipelineLayout().get(), 0U,
      descriptor_set.getDescriptorSet().get(), {});
  m_commandBuffer.dispatch(work_group_size.x, work_group_size.y,
                           work_group_size.z);
}
