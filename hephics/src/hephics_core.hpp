/*
This header file is part of the Hephics project.
This header file is used to manage gpu commands.
*/

#pragma once

#include "hephics_core/gpu.hpp"
#include "hephics_core/io.hpp"
#include "hephics_core/ui.hpp"

namespace hpxc {

struct ComputeWorkGroupSize {
  uint32_t x;
  uint32_t y;
  uint32_t z;
};

struct CommandBeginInfo {
  vk::CommandBufferUsageFlags usage_flags =
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  uint32_t subpass_index = 0U;

  CommandBeginInfo() = default;
};

/// <summary>
/// This class is interface for gpu commands.
/// This is base (not abstract).
/// </summary>
class CommandBuffer {
 protected:
  friend class CommandDriver;

  vk::CommandBuffer m_commandBuffer;
  bool m_isSecondary = false;

  CommandBuffer(const vk::UniqueCommandBuffer& ptr_command_buffer,
                const bool is_secondary = false)
      : m_commandBuffer(ptr_command_buffer.get()),
        m_isSecondary(is_secondary) {}

 public:
  /// <summary>
  /// Begin gpu commands.
  /// This function must be used.
  /// </summary>
  /// <param name="command_begin_info"></param>
  void begin(const CommandBeginInfo& command_begin_info = {}) const;

  /// <summary>
  /// End gpu commands.
  /// This function must be used.
  /// </summary>
  void end() const;

  /// <summary>
  /// Set buffer access barrier.
  /// </summary>
  /// <param name="barrier">buffer barrier</param>
  /// <param name="src_stage">priority stage</param>
  /// <param name="dst_stage">wait(next) stage</param>
  void setPipelineBarrier(const gpu::BufferBarrier& barrier,
                          const PipelineStage src_stage,
                          const PipelineStage dst_stage) const;

  /// <summary>
  /// Set image access barrier.
  /// </summary>
  /// <param name="barrier">image barrier</param>
  /// <param name="src_stage">priority stage</param>
  /// <param name="dst_stage">wait(next) stage</param>
  void setPipelineBarrier(const gpu::ImageBarrier& barrier,
                          const PipelineStage src_stage,
                          const PipelineStage dst_stage) const;

  /// <summary>
  /// Register push constants.
  /// </summary>
  /// <param name="pipeline"></param>
  /// <param name="dst_stages">shader stages using push constants</param>
  /// <param name="offset"></param>
  /// <param name="data">push constants content data</param>
  void pushConstants(const gpu::Pipeline& pipeline,
                     const std::vector<ShaderStage>& dst_stages,
                     const uint32_t offset,
                     const std::vector<float_t>& data) const;

  /// <summary>
  /// Reset gpu commands.
  /// </summary>
  void resetCommands() const;
};

/// <summary>
/// This class is interface for gpu transfer commands.
/// </summary>
class TransferCommandBuffer : public CommandBuffer {
 protected:
  friend class CommandDriver;

  TransferCommandBuffer(const vk::UniqueCommandBuffer& ptr_command_buffer,
                        const bool is_secondary = false)
      : CommandBuffer(ptr_command_buffer, is_secondary) {}

 public:
  /// <summary>
  /// Copy buffer data to buffer.
  ///   cpu staging buffer <-> gpu device buffer
  /// </summary>
  /// <param name="staging_buffer">cpu buffer having data</param>
  /// <param name="dst_buffer">gpu local buffer</param>
  void copyBuffer(const gpu::Buffer& staging_buffer,
                  const gpu::Buffer& dst_buffer) const;

  /// <summary>
  /// Copy cpu staging buffer data to gpu image.
  /// </summary>
  /// <param name="buffer">cpu buffer having image data</param>
  /// <param name="image">gpu local image</param>
  /// <param name="image_layout">
  ///   gpu image layout
  ///     (TransferDstOptimal recomended)
  /// </param>
  /// <param name="image_view_info"></param>
  void copyBufferToImage(const gpu::Buffer& buffer, const gpu::Image& image,
                         const ImageLayout image_layout,
                         const ImageViewInfo& image_view_info) const;

  /// <summary>
  /// Copy gpu image data to cpu staging buffer.
  /// </summary>
  /// <param name="image">gpu local image</param>
  /// <param name="buffer">cpu staging buffer</param>
  /// <param name="image_layout">
  ///   gpu image layout
  ///     (TransferSrcOptimal recomended)
  /// </param>
  /// <param name="image_view_info"></param>
  void copyImageToBuffer(const gpu::Image& image, const gpu::Buffer& buffer,
                         const ImageLayout image_layout,
                         const ImageViewInfo& image_view_info) const;

  /// <summary>
  /// Set mipmaps to gpu image.
  /// </summary>
  /// <param name="image">image associate with mipmaps</param>
  /// <param name="dst_stage"></param>
  void setMipmaps(const gpu::Image& image, const PipelineStage dst_stage) const;

  /// <summary>
  /// Transfer mipmaps image ownership into another queue family.
  /// Src owner command buffer must calls this function.
  /// </summary>
  /// <param name="image">mipmaps image</param>
  /// <param name="src_stage"></param>
  /// <param name="dst_stage"></param>
  /// <param name="queue_family_index">first: src, second: dst</param>
  void transferMipmapImages(
      const gpu::Image& image, const PipelineStage src_stage,
      const PipelineStage dst_stage,
      std::pair<uint32_t, uint32_t> queue_family_index) const;

  /// <summary>
  /// Acquire mipmaps image ownership from another queue family.
  /// Dst owner command buffer must calls this function.
  /// </summary>
  /// <param name="image"></param>
  /// <param name="src_stage"></param>
  /// <param name="dst_stage"></param>
  /// <param name="queue_family_index">first: src, second: dst</param>
  void acquireMipmapImages(
      const gpu::Image& image, const PipelineStage src_stage,
      const PipelineStage dst_stage,
      std::pair<uint32_t, uint32_t> queue_family_index) const;
};

/// <summary>
/// This class is interface for gpu compute commands.
/// </summary>
class ComputeCommandBuffer : public TransferCommandBuffer {
 protected:
  friend class CommandDriver;

  ComputeCommandBuffer(const vk::UniqueCommandBuffer& ptr_command_buffer,
                       const bool is_secondary = false)
      : TransferCommandBuffer(ptr_command_buffer, is_secondary) {}

 public:
  /// <summary>
  /// Execute compute shader.
  /// </summary>
  /// <param name="pipeline">constructed as compute pipeline</param>
  /// <param name="descriptor_set"></param>
  /// <param name="work_group_size">
  /// Triple number considered from
  ///   resource size and local_size in shader
  /// </param>
  void compute(const gpu::Pipeline& pipeline,
               const gpu::DescriptorSet& descriptor_set,
               const ComputeWorkGroupSize& work_group_size) const;
};

/// <summary>
/// This class is interface for gpu graphic commands.
/// </summary>
class GraphicCommandBuffer : public ComputeCommandBuffer {
 protected:
  friend class CommandDriver;

  GraphicCommandBuffer(const vk::UniqueCommandBuffer& ptr_command_buffer,
                       const bool is_secondary = false)
      : ComputeCommandBuffer(ptr_command_buffer, is_secondary) {}

 public:
};

/// <summary>
/// This class provide hpxc::CommandBuffer family.
/// This class has vulkan's commandbuffer interface substance.
/// And, this class enables you multiple threads adding gpu commands.
/// </summary>
class CommandDriver {
 private:
  vk::Queue m_queue;
  vk::UniqueCommandPool m_ptrCommandPool;
  vk::UniqueCommandBuffer m_ptrPrimaryCommandBuffer;
  std::vector<vk::UniqueCommandPool> m_ptrSecondaryCommandPools;
  std::vector<vk::UniqueCommandBuffer> m_secondaryCommandBuffers;

  QueueFamilyType m_queueFamilyType;
  uint32_t m_queueFamilyIndex;

 public:
  CommandDriver(const std::unique_ptr<gpu::Context>& ptr_context,
                const hpxc::QueueFamilyType queue_family);
  ~CommandDriver();

  void destroySecondary() { m_secondaryCommandBuffers.clear(); }

  /// <summary>
  /// Allocate secondary command buffers.
  /// Secondary command buffers are used for multi-threading.
  /// </summary>
  /// <param name="ptr_context"></param>
  /// <param name="required_secondary_num"></param>
  void constructSecondary(const std::unique_ptr<gpu::Context>& ptr_context,
                          const uint32_t required_secondary_num = 1U);

  void resetAllCommands() const;
  void resetAllCommandPools(
      const std::unique_ptr<gpu::Context>& ptr_context) const;

  /// <summary>
  /// Integrate secondary command into primary command buffer.
  /// If secondary is used,
  ///   this function must be called before primary's end() command.
  /// </summary>
  void mergeSecondaryCommands() const;

  /// <summary>
  /// Submit gpu commands.
  /// </summary>
  /// <param name="wait_stage"></param>
  /// <param name="semaphore"></param>
  void submit(const PipelineStage wait_stage, gpu::Semaphore& semaphore) const;

  CommandBuffer getPrimary() const {
    return CommandBuffer(m_ptrPrimaryCommandBuffer);
  }
  ComputeCommandBuffer getCompute(
      const std::optional<size_t> secondary_index = std::nullopt) const;
  TransferCommandBuffer getTransfer(
      const std::optional<size_t> secondary_index = std::nullopt) const;

  const auto& getQueueFamilyType() const { return m_queueFamilyType; }
  const auto& getQueueFamilyIndex() const { return m_queueFamilyIndex; }
};

gpu::Buffer createStagingBufferToGPU(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size);
gpu::Buffer* createPtrStagingBufferToGPU(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size);

gpu::Buffer createStagingBufferFromGPU(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size);
gpu::Buffer* createPtrStagingBufferFromGPU(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size);

gpu::Buffer createStorageBuffer(
    const std::unique_ptr<gpu::Context>& ptr_context,
    TransferType transfer_type, const size_t size);
gpu::Buffer* createPtrStorageBuffer(
    const std::unique_ptr<gpu::Context>& ptr_context,
    TransferType transfer_type, const size_t size);

gpu::Buffer createUniformBuffer(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size);
gpu::Buffer* createPtrUniformBuffer(
    const std::unique_ptr<gpu::Context>& ptr_context, const size_t size);

using ShaderModuleMap = std::unordered_map<std::string, gpu::ShaderModule>;

}  // namespace hpxc
