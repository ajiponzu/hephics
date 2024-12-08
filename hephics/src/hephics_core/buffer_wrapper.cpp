#include "../hephics_core.hpp"

hpxc::gpu::Buffer hpxc::createStagingBufferToGPU(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size) {
  return gpu::Buffer(ptr_context, MemoryUsage::CpuToGpu,
                     TransferType::TransferSrc, {BufferUsage::StagingBuffer},
                     size);
}

hpxc::gpu::Buffer* hpxc::createPtrStagingBufferToGPU(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size) {
  return new gpu::Buffer(ptr_context, MemoryUsage::CpuToGpu,
                         TransferType::TransferSrc,
                         {BufferUsage::StagingBuffer}, size);
}

hpxc::gpu::Buffer hpxc::createStagingBufferFromGPU(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size) {
  return gpu::Buffer(ptr_context, MemoryUsage::GpuToCpu,
                     TransferType::TransferDst, {BufferUsage::StagingBuffer},
                     size);
}

hpxc::gpu::Buffer* hpxc::createPtrStagingBufferFromGPU(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size) {
  return new gpu::Buffer(ptr_context, MemoryUsage::GpuToCpu,
                         TransferType::TransferDst,
                         {BufferUsage::StagingBuffer}, size);
}

hpxc::gpu::Buffer hpxc::createStorageBuffer(
    const std::unique_ptr<gpu::Context>& ptr_context,
    TransferType transfer_type, const size_t size) {
  return gpu::Buffer(ptr_context, MemoryUsage::GpuOnly, transfer_type,
                     {BufferUsage::StorageBuffer}, size);
}

hpxc::gpu::Buffer* hpxc::createPtrStorageBuffer(
    const std::unique_ptr<gpu::Context>& ptr_context,
    TransferType transfer_type, const size_t size) {
  return new gpu::Buffer(ptr_context, MemoryUsage::GpuOnly, transfer_type,
                         {BufferUsage::StorageBuffer}, size);
}

hpxc::gpu::Buffer hpxc::createUniformBuffer(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size) {
  return gpu::Buffer(ptr_context, MemoryUsage::CpuToGpu,
                     TransferType::TransferDst, {BufferUsage::UniformBuffer},
                     size);
}

hpxc::gpu::Buffer* hpxc::createPtrUniformBuffer(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size) {
  return new gpu::Buffer(ptr_context, MemoryUsage::CpuToGpu,
                         TransferType::TransferDst,
                         {BufferUsage::UniformBuffer}, size);
}
