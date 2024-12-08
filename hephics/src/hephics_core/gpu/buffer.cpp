#include <iomanip>
#include <iostream>

#include "../gpu.hpp"
#include "vk_helper.hpp"

static vk::BufferUsageFlags get_transfer_usage_flags(
    const hpxc::TransferType transfer_type) {
  using TransferType = hpxc::TransferType;

  switch (transfer_type) {
    case TransferType::TransferSrc:
      return vk::BufferUsageFlagBits::eTransferSrc;
    case TransferType::TransferDst:
      return vk::BufferUsageFlagBits::eTransferDst;
    case TransferType::TransferSrcDst:
      return vk::BufferUsageFlagBits::eTransferSrc |
             vk::BufferUsageFlagBits::eTransferDst;
    default:
      return vk::BufferUsageFlagBits::eTransferSrc;
  }
}

static vk::BufferUsageFlagBits get_buffer_usage(
    const hpxc::BufferUsage buffer_usage) {
  using BufferUsage = hpxc::BufferUsage;

  switch (buffer_usage) {
    case BufferUsage::VertexBuffer:
      return vk::BufferUsageFlagBits::eVertexBuffer;
    case BufferUsage::IndexBuffer:
      return vk::BufferUsageFlagBits::eIndexBuffer;
    case BufferUsage::UniformBuffer:
      return vk::BufferUsageFlagBits::eUniformBuffer;
    case BufferUsage::StorageBuffer:
      return vk::BufferUsageFlagBits::eStorageBuffer;
    case BufferUsage::StagingBuffer:
      return vk::BufferUsageFlagBits::eTransferSrc;
    default:
      return vk::BufferUsageFlagBits::eVertexBuffer;
  }
}

hpxc::gpu::Buffer::Buffer(const std::unique_ptr<Context>& ptr_context,
                                  const MemoryUsage memory_usage,
                                  const TransferType transfer_type,
                                  const std::vector<BufferUsage>& buffer_usages,
                                  const size_t size)
    : m_size(size) {
  {
    const vk::BufferUsageFlags vk_transfer_type =
        get_transfer_usage_flags(transfer_type);

    vk::BufferUsageFlags vk_buffer_usages{};
    for (const auto& buffer_usage : buffer_usages) {
      vk_buffer_usages |= get_buffer_usage(buffer_usage);
    }

    vk::BufferCreateInfo buffer_info{};
    buffer_info.setUsage(vk_transfer_type | vk_buffer_usages);
    buffer_info.setSize(m_size);
    buffer_info.setSharingMode(vk::SharingMode::eExclusive);

    m_ptrBuffer =
        ptr_context->getDevice()->getLogicalDevice()->createBufferUnique(
            buffer_info);
  }

  {
    const auto memory_requirements =
        ptr_context->getDevice()
            ->getLogicalDevice()
            ->getBufferMemoryRequirements(m_ptrBuffer.get());

    const vk::MemoryPropertyFlags vk_memory_usage =
        vk_helper::getMemoryPropertyFlags(memory_usage);

    const auto memory_props =
        ptr_context->getDevice()->getPhysicalDevice().getMemoryProperties();

    uint32_t memory_type_idx = 0U;
    for (; memory_type_idx < memory_props.memoryTypeCount;
         memory_type_idx += 1U) {
      if ((memory_requirements.memoryTypeBits & (1 << memory_type_idx)) &&
          (memory_props.memoryTypes.at(memory_type_idx).propertyFlags &
           vk_memory_usage) == vk_memory_usage) {
        break;
      }
    }

    vk::MemoryAllocateInfo allocation_info{};
    allocation_info.setMemoryTypeIndex(memory_type_idx);
    allocation_info.setAllocationSize(memory_requirements.size);

    m_ptrMemory =
        ptr_context->getDevice()->getLogicalDevice()->allocateMemoryUnique(
            allocation_info);
  }

  ptr_context->getDevice()->getLogicalDevice()->bindBufferMemory(
      m_ptrBuffer.get(), m_ptrMemory.get(), 0);
}

hpxc::gpu::Buffer::~Buffer() {}

void* hpxc::gpu::Buffer::mapMemory(
    const std::unique_ptr<Context>& ptr_context) const {
  return ptr_context->getDevice()->getLogicalDevice()->mapMemory(
      m_ptrMemory.get(), 0, m_size, {});
}

void hpxc::gpu::Buffer::unmapMemory(
    const std::unique_ptr<Context>& ptr_context) const {
  ptr_context->getDevice()->getLogicalDevice()->unmapMemory(m_ptrMemory.get());
}
