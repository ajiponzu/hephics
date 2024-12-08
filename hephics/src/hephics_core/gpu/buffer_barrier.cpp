#include "../gpu.hpp"
#include "vk_helper.hpp"

hpxc::gpu::BufferBarrier::BufferBarrier(
    const Buffer& buffer, const std::vector<AccessFlag>& priority_access_flags,
    const std::vector<AccessFlag> wait_access_flags) {
  m_bufferMemoryBarrier.setBuffer(buffer.getBuffer());
  m_bufferMemoryBarrier.setSize(buffer.getSize());

  {
    vk::AccessFlags access_flags;
    for (const auto& access_flag : priority_access_flags) {
      access_flags |= vk_helper::getAccessFlagBits(access_flag);
    }

    m_bufferMemoryBarrier.setSrcAccessMask(access_flags);
  }

  {
    vk::AccessFlags access_flags;
    for (const auto& access_flag : wait_access_flags) {
      access_flags |= vk_helper::getAccessFlagBits(access_flag);
    }
    m_bufferMemoryBarrier.setDstAccessMask(access_flags);
  }
}

hpxc::gpu::BufferBarrier::~BufferBarrier() {}
