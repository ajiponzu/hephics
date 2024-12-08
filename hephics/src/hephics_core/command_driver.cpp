#include <iostream>

#include "../hephics_core.hpp"
#include "gpu/vk_helper.hpp"

hpxc::CommandDriver::CommandDriver(
    const std::unique_ptr<gpu::Context>& ptr_context,
    const hpxc::QueueFamilyType queue_family) {
  m_queueFamilyType = queue_family;
  m_queueFamilyIndex =
      ptr_context->getDevice()->getQueueFamilyIndex(queue_family);

  m_queue = ptr_context->getDevice()->getQueue(m_queueFamilyIndex);

  {
    vk::CommandPoolCreateInfo pool_info{{}, m_queueFamilyIndex};
    pool_info.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    m_ptrCommandPool =
        ptr_context->getDevice()->getLogicalDevice()->createCommandPoolUnique(
            pool_info);
  }

  vk::CommandBufferAllocateInfo alloc_info{
      m_ptrCommandPool.get(), vk::CommandBufferLevel::ePrimary, 1U};

  m_ptrPrimaryCommandBuffer =
      std::move(ptr_context->getDevice()
                    ->getLogicalDevice()
                    ->allocateCommandBuffersUnique(alloc_info)
                    .front());
}

hpxc::CommandDriver::~CommandDriver() {}

void hpxc::CommandDriver::constructSecondary(
    const std::unique_ptr<gpu::Context>& ptr_context,
    const uint32_t required_secondary_num) {
  for (uint32_t idx = 0U; idx < required_secondary_num; idx += 1U) {
    {
      vk::CommandPoolCreateInfo pool_info{{}, m_queueFamilyIndex};
      pool_info.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

      m_ptrSecondaryCommandPools.push_back(
          ptr_context->getDevice()->getLogicalDevice()->createCommandPoolUnique(
              pool_info));
    }

    vk::CommandBufferAllocateInfo alloc_info{
        m_ptrSecondaryCommandPools.back().get(),
        vk::CommandBufferLevel::eSecondary, 1U};

    m_secondaryCommandBuffers.push_back(
        std::move(ptr_context->getDevice()
                      ->getLogicalDevice()
                      ->allocateCommandBuffersUnique(alloc_info)
                      .front()));
  }
}

void hpxc::CommandDriver::resetAllCommands() const {
  for (size_t idx = 0U; idx < m_secondaryCommandBuffers.size(); idx += 1U) {
    const auto& command_buffer = m_secondaryCommandBuffers.at(idx);

    command_buffer->reset(vk::CommandBufferResetFlags());
  }

  m_ptrPrimaryCommandBuffer->reset(vk::CommandBufferResetFlags());
}

void hpxc::CommandDriver::resetAllCommandPools(
    const std::unique_ptr<gpu::Context>& ptr_context) const {
  for (size_t idx = 0U; idx < m_ptrSecondaryCommandPools.size(); idx += 1U) {
    ptr_context->getDevice()->getLogicalDevice()->resetCommandPool(
        m_ptrSecondaryCommandPools.at(idx).get(), vk::CommandPoolResetFlags());
  }

  ptr_context->getDevice()->getLogicalDevice()->resetCommandPool(
      m_ptrCommandPool.get(), vk::CommandPoolResetFlags());
}

void hpxc::CommandDriver::mergeSecondaryCommands() const {
  std::vector<vk::CommandBuffer> command_buffers;
  for (const auto& command_buffer : m_secondaryCommandBuffers) {
    command_buffers.push_back(command_buffer.get());
  }

  m_ptrPrimaryCommandBuffer->executeCommands(command_buffers);
}

void hpxc::CommandDriver::submit(const PipelineStage wait_stage,
                                 gpu::Semaphore& semaphore) const {
  vk::SubmitInfo submit_info;
  submit_info.setPNext(semaphore.getPtrTimelineSubmitInfo());
  submit_info.setCommandBuffers(m_ptrPrimaryCommandBuffer.get());
  submit_info.setSignalSemaphores(semaphore.getSemaphore().get());
  submit_info.setWaitSemaphores(semaphore.getSemaphore().get());

  semaphore.setWaitStage(vk_helper::getPipelineStageFlagBits(wait_stage));
  submit_info.setWaitDstStageMask(semaphore.getBackWaitStage());

  m_queue.submit(submit_info);

  semaphore.updateWaitValue();
  semaphore.updateSignalValue();
}

hpxc::ComputeCommandBuffer hpxc::CommandDriver::getCompute(
    const std::optional<size_t> secondary_index) const {
  if (secondary_index.has_value()) {
    return ComputeCommandBuffer(
        m_secondaryCommandBuffers.at(secondary_index.value()), true);
  }

  return ComputeCommandBuffer(m_ptrPrimaryCommandBuffer);
}

hpxc::TransferCommandBuffer hpxc::CommandDriver::getTransfer(
    const std::optional<size_t> secondary_index) const {
  if (secondary_index.has_value()) {
    return TransferCommandBuffer(
        m_secondaryCommandBuffers.at(secondary_index.value()), true);
  }

  return TransferCommandBuffer(m_ptrPrimaryCommandBuffer);
}
