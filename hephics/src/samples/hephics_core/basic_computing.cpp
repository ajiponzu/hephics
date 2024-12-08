#include "basic_computing.hpp"

#include <iostream>
#include <shared_mutex>
#include <thread>

static void set_transfer_secondary_command(
    const hpxc::TransferCommandBuffer& command_buffer,
    const std::unique_ptr<hpxc::gpu::Buffer>& transfered_buffer,
    const std::pair<uint32_t, uint32_t> queue_family_indices,
    hpxc::gpu::Buffer& staging_buffer) {
  command_buffer.begin();

  command_buffer.copyBuffer(staging_buffer, *transfered_buffer);

  // release the ownership of the gpu storage buffer
  auto buffer_barrier = hpxc::gpu::BufferBarrier(
      *transfered_buffer, {hpxc::AccessFlag::TransferWrite},
      {hpxc::AccessFlag::ShaderRead, hpxc::AccessFlag::ShaderWrite});
  buffer_barrier.setSrcQueueFamilyIndex(queue_family_indices.first);
  buffer_barrier.setDstQueueFamilyIndex(queue_family_indices.second);

  command_buffer.setPipelineBarrier(buffer_barrier,
                                    hpxc::PipelineStage::Transfer,
                                    hpxc::PipelineStage::BottomOfPipe);

  command_buffer.end();
}

samples::core::BasicComputing::BasicComputing() {
  m_ptrContext = std::make_unique<hpxc::gpu::Context>(nullptr);

  m_ptrComputeCommandDriver.reset(
      new hpxc::CommandDriver(m_ptrContext, hpxc::QueueFamilyType::Compute));

  m_ptrTransferCommandDriver.reset(
      new hpxc::CommandDriver(m_ptrContext, hpxc::QueueFamilyType::Transfer));

  m_ptrUniformBuffer.reset(
      hpxc::createPtrUniformBuffer(m_ptrContext, sizeof(float_t)));
  const auto uniform_mapped_address =
      m_ptrUniformBuffer->mapMemory(m_ptrContext);
  std::fill_n(reinterpret_cast<float_t*>(uniform_mapped_address),
              m_ptrUniformBuffer->getSize() / sizeof(float_t), 3.14f);
  m_ptrUniformBuffer->unmapMemory(m_ptrContext);

  m_ptrInputStorageBuffer.reset(hpxc::createPtrStorageBuffer(
      m_ptrContext, hpxc::TransferType::TransferDst, sizeof(uint32_t) * 1024U));

  m_ptrOutputStorageBuffer.reset(hpxc::createPtrStorageBuffer(
      m_ptrContext, hpxc::TransferType::TransferSrcDst,
      sizeof(uint32_t) * 1024U));
}

samples::core::BasicComputing::~BasicComputing() {
  m_ptrContext->getDevice()->waitIdle();
}

void samples::core::BasicComputing::run() {
  auto result_buffer = hpxc::createStagingBufferFromGPU(
      m_ptrContext, m_ptrOutputStorageBuffer->getSize());

  {
    std::vector<hpxc::gpu::Buffer> staging_buffers;

    constructShaderResources();
    setTransferCommands(staging_buffers);
    setComputeCommands(result_buffer);

    hpxc::gpu::Semaphore semaphore(m_ptrContext);
    m_ptrTransferCommandDriver->submit(hpxc::PipelineStage::BottomOfPipe,
                                       semaphore);
    m_ptrComputeCommandDriver->submit(hpxc::PipelineStage::Transfer, semaphore);
    semaphore.wait(m_ptrContext);
  }

  {
    hpxc::gpu::Semaphore semaphore(m_ptrContext);

    std::vector<uint32_t> result(result_buffer.getSize() / sizeof(uint32_t));
    const auto result_mapped_address = result_buffer.mapMemory(m_ptrContext);
    std::memcpy(reinterpret_cast<void*>(result.data()), result_mapped_address,
                result_buffer.getSize());
    result_buffer.unmapMemory(m_ptrContext);

    for (size_t idx = 0U; idx < result.size(); idx += 1U) {
      std::cout << "idx[" << idx << "]: " << result[idx] << std::endl;
    }
  }

  m_ptrComputeCommandDriver->resetAllCommandPools(m_ptrContext);
  m_ptrTransferCommandDriver->resetAllCommandPools(m_ptrContext);
}

void samples::core::BasicComputing::setTransferCommands(
    std::vector<hpxc::gpu::Buffer>& staging_buffers) {
  std::shared_mutex mutex;

  m_ptrTransferCommandDriver->constructSecondary(m_ptrContext, 2);

  staging_buffers.push_back(hpxc::createStagingBufferToGPU(
      m_ptrContext, m_ptrInputStorageBuffer->getSize()));
  staging_buffers.push_back(hpxc::createStagingBufferToGPU(
      m_ptrContext, m_ptrOutputStorageBuffer->getSize()));

  const auto src_queue_family_index =
      m_ptrContext->getDevice()->getQueueFamilyIndex(
          hpxc::QueueFamilyType::Transfer);
  const auto dst_queue_family_index =
      m_ptrContext->getDevice()->getQueueFamilyIndex(
          hpxc::QueueFamilyType::Compute);

  // multi-threading
  std::thread transfer_thread0([&]() {
    const size_t thread_index = 0U;

    std::shared_lock lock(mutex);
    const auto command_buffer =
        m_ptrTransferCommandDriver->getTransfer(thread_index);
    lock.unlock();

    lock.lock();
    auto& staging_buffer = staging_buffers[thread_index];
    const auto mapped_address = staging_buffer.mapMemory(m_ptrContext);
    lock.unlock();

    std::fill_n(reinterpret_cast<uint32_t*>(mapped_address),
                staging_buffer.getSize() / sizeof(uint32_t), 5U);

    lock.lock();
    staging_buffer.unmapMemory(m_ptrContext);
    lock.unlock();

    set_transfer_secondary_command(
        command_buffer, m_ptrInputStorageBuffer,
        {src_queue_family_index, dst_queue_family_index}, staging_buffer);
  });

  std::thread transfer_thread1([&]() {
    const size_t thread_index = 1U;

    std::shared_lock lock(mutex);
    const auto command_buffer =
        m_ptrTransferCommandDriver->getTransfer(thread_index);
    lock.unlock();

    lock.lock();
    auto& staging_buffer = staging_buffers[thread_index];
    const auto mapped_address = staging_buffer.mapMemory(m_ptrContext);
    lock.unlock();

    std::fill_n(reinterpret_cast<uint32_t*>(mapped_address),
                staging_buffer.getSize() / sizeof(uint32_t), 5U);

    lock.lock();
    staging_buffer.unmapMemory(m_ptrContext);
    lock.unlock();

    set_transfer_secondary_command(
        command_buffer, m_ptrOutputStorageBuffer,
        {src_queue_family_index, dst_queue_family_index}, staging_buffer);
  });

  const auto primary_command_buffer = m_ptrTransferCommandDriver->getPrimary();

  primary_command_buffer.begin();

  // wait for the completion of the secondary command
  transfer_thread0.join();
  transfer_thread1.join();

  // merge secondary command
  m_ptrTransferCommandDriver->mergeSecondaryCommands();

  primary_command_buffer.end();
}

void samples::core::BasicComputing::constructShaderResources() {
  const auto spirv_binary =
      hpxc::io::shader::read("shaders/compute/basic.comp");

  m_shaderModuleMap["compute"] =
      hpxc::gpu::ShaderModule(m_ptrContext, spirv_binary);

  const auto description_unit =
      hpxc::gpu::DescriptionUnit(m_shaderModuleMap, {"compute"});

  m_ptrDescriptorSetLayout.reset(
      new hpxc::gpu::DescriptorSetLayout(m_ptrContext, description_unit));

  m_ptrDescriptorSet.reset(
      new hpxc::gpu::DescriptorSet(m_ptrContext, *m_ptrDescriptorSetLayout));

  std::vector<hpxc::gpu::BufferDescription> buffer_descriptions;
  buffer_descriptions.emplace_back(
      description_unit.getDescriptorInfoMap().at("UniformNumber"),
      *m_ptrUniformBuffer);
  buffer_descriptions.emplace_back(
      description_unit.getDescriptorInfoMap().at("Output"),
      *m_ptrOutputStorageBuffer);
  buffer_descriptions.emplace_back(
      description_unit.getDescriptorInfoMap().at("Input"),
      *m_ptrInputStorageBuffer);

  m_ptrDescriptorSet->updateDescriptorSet(m_ptrContext, buffer_descriptions,
                                          {});

  m_ptrComputePipeline.reset(new hpxc::gpu::Pipeline(
      m_ptrContext, description_unit, *m_ptrDescriptorSetLayout));
  m_ptrComputePipeline->constructComputePipeline(
      m_ptrContext, m_shaderModuleMap.at("compute"));
}

void samples::core::BasicComputing::setComputeCommands(
    hpxc::gpu::Buffer& staging_buffer) {
  const auto command_buffer = m_ptrComputeCommandDriver->getCompute();

  command_buffer.begin();

  {
    // acquire the ownership of the gpu storage buffer
    // acrquire barrier parameters are same as the release barrier.
    auto buffer_barrier = hpxc::gpu::BufferBarrier(
        *m_ptrInputStorageBuffer, {hpxc::AccessFlag::TransferWrite},
        {hpxc::AccessFlag::ShaderRead, hpxc::AccessFlag::ShaderWrite});
    buffer_barrier.setSrcQueueFamilyIndex(
        m_ptrTransferCommandDriver->getQueueFamilyIndex());
    buffer_barrier.setDstQueueFamilyIndex(
        m_ptrComputeCommandDriver->getQueueFamilyIndex());

    command_buffer.setPipelineBarrier(buffer_barrier,
                                      hpxc::PipelineStage::BottomOfPipe,
                                      hpxc::PipelineStage::ComputeShader);
  }

  {
    // acquire the ownership of the gpu storage buffer
    auto buffer_barrier = hpxc::gpu::BufferBarrier(
        *m_ptrOutputStorageBuffer, {hpxc::AccessFlag::TransferWrite},
        {hpxc::AccessFlag::ShaderRead, hpxc::AccessFlag::ShaderWrite});
    buffer_barrier.setSrcQueueFamilyIndex(
        m_ptrTransferCommandDriver->getQueueFamilyIndex());
    buffer_barrier.setDstQueueFamilyIndex(
        m_ptrComputeCommandDriver->getQueueFamilyIndex());

    command_buffer.setPipelineBarrier(buffer_barrier,
                                      hpxc::PipelineStage::BottomOfPipe,
                                      hpxc::PipelineStage::ComputeShader);
  }

  command_buffer.compute(*m_ptrComputePipeline, *m_ptrDescriptorSet,
                         hpxc::ComputeWorkGroupSize{4U, 1U, 1U});

  {
    const auto buffer_barrier = hpxc::gpu::BufferBarrier(
        *m_ptrOutputStorageBuffer,
        {hpxc::AccessFlag::ShaderRead, hpxc::AccessFlag::ShaderWrite},
        {hpxc::AccessFlag::TransferRead});

    command_buffer.setPipelineBarrier(buffer_barrier,
                                      hpxc::PipelineStage::ComputeShader,
                                      hpxc::PipelineStage::Transfer);
  }

  command_buffer.copyBuffer(*m_ptrOutputStorageBuffer, staging_buffer);

  {
    const auto buffer_barrier = hpxc::gpu::BufferBarrier(
        *m_ptrOutputStorageBuffer, {hpxc::AccessFlag::TransferRead},
        {hpxc::AccessFlag::Unknown});

    command_buffer.setPipelineBarrier(buffer_barrier,
                                      hpxc::PipelineStage::Transfer,
                                      hpxc::PipelineStage::BottomOfPipe);
  }

  command_buffer.end();
}
