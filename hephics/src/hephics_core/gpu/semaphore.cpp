#include <iostream>

#include "../gpu.hpp"

hpxc::gpu::Semaphore::Semaphore(const std::unique_ptr<Context>& ptr_context) {
  {
    vk::SemaphoreTypeCreateInfo semaphore_type_info;
    semaphore_type_info.setSemaphoreType(vk::SemaphoreType::eTimeline);

    vk::SemaphoreCreateInfo semaphore_info;
    semaphore_info.setPNext(&semaphore_type_info);

    m_ptrSemaphore =
        ptr_context->getDevice()->getLogicalDevice()->createSemaphoreUnique(
            semaphore_info);
  }

  m_timelineSubmitInfo.setWaitSemaphoreValues(m_waitValue);
  m_timelineSubmitInfo.setSignalSemaphoreValues(m_signalValue);
}

hpxc::gpu::Semaphore::~Semaphore() {}

void hpxc::gpu::Semaphore::wait(const std::unique_ptr<Context>& ptr_context) {
  vk::SemaphoreWaitInfo semaphore_wait_info;
  semaphore_wait_info.setSemaphores(m_ptrSemaphore.get());
  semaphore_wait_info.setValues(m_waitValue);

  const auto vk_result =
      ptr_context->getDevice()->getLogicalDevice()->waitSemaphores(
          semaphore_wait_info, std::numeric_limits<uint64_t>::max());

  if (vk_result != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to wait for semaphore");
  }

  // Recycle this semaphore values
  m_signalValue = 1U;
  m_waitValue = 0U;
  m_waitStages.clear();

  {
    vk::SemaphoreTypeCreateInfo semaphore_type_info;
    semaphore_type_info.setSemaphoreType(vk::SemaphoreType::eTimeline);

    vk::SemaphoreCreateInfo semaphore_info;
    semaphore_info.setPNext(&semaphore_type_info);

    m_ptrSemaphore =
        ptr_context->getDevice()->getLogicalDevice()->createSemaphoreUnique(
            semaphore_info);
  }

  m_timelineSubmitInfo = vk::TimelineSemaphoreSubmitInfoKHR{};
  m_timelineSubmitInfo.setWaitSemaphoreValues(m_waitValue);
  m_timelineSubmitInfo.setSignalSemaphoreValues(m_signalValue);
}
