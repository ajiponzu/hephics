/*
This header file is part of the Hephics project.
Especially, the content of this file is a driver for Vulkan API's each moudle.
*/

#pragma once

#ifdef _DEBUG
#define HEPHICS_DEBUG
#endif

#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <variant>

#include "module_connection/gpu_ui.hpp"

namespace hpxc {

enum class QueueFamilyType {
  Graphics = 0U,
  Compute,
  Transfer,
};

enum class MemoryUsage {
  Unknown = 0U,
  GpuOnly,
  CpuOnly,
  CpuToGpu,
  GpuToCpu,
};

enum class TransferType {
  Unknown = 0U,
  TransferSrc,
  TransferDst,
  TransferSrcDst,
};

enum class BufferUsage {
  Unknown = 0U,
  VertexBuffer,
  IndexBuffer,
  UniformBuffer,
  StorageBuffer,
  StagingBuffer,
};

enum class ImageUsage {
  Unknown = 0U,
  Sampled,
  Storage,
  ColorAttachment,
  DepthStencilAttachment,
  TransientAttachment,
  InputAttachment,
};

enum class ImageFormat {
  Unknown = 0U,
  R8Unorm,
  R8Snorm,
  R8Uscaled,
  R8Sscaled,
  R8Uint,
  R8Sint,
  R8Srgb,
  R8G8Unorm,
  R8G8Snorm,
  R8G8Uscaled,
  R8G8Sscaled,
  R8G8Uint,
  R8G8Sint,
  R8G8Srgb,
  R8G8B8Unorm,
  R8G8B8Snorm,
  R8G8B8Uscaled,
  R8G8B8Sscaled,
  R8G8B8Uint,
  R8G8B8Sint,
  R8G8B8Srgb,
  R8G8B8A8Unorm,
  R8G8B8A8Snorm,
  R8G8B8A8Uscaled,
  R8G8B8A8Sscaled,
  R8G8B8A8Uint,
  R8G8B8A8Sint,
  R8G8B8A8Srgb,
};

enum class ImageDimension {
  Unknown = 0U,
  v1D,
  v2D,
  v3D,
};

enum class ImageSampleCount {
  Unknown = 0U,
  v1,
  v2,
  v4,
  v8,
  v16,
  v32,
  v64,
};

enum class ShaderType {
  Vertex = 0U,
  Fragment,
  Compute,
};

enum class ImageAspect {
  Unknown = 0U,
  Color,
  Depth,
  Stencil,
  DepthStencil,
};

enum class SamplerFilter {
  Linear = 0U,
  Nearest,
  Cubic,
};

enum class SamplerMipmapMode {
  Nearest = 0U,
  Linear,
};

enum class SamplerAddressMode {
  Repeat = 0U,
  MirroredRepeat,
  ClampToEdge,
  ClampToBorder,
  MirrorClampToEdge,
};

enum class SamplerBorderColor {
  FloatTransparentBlack = 0U,
  IntTransparentBlack,
  FloatOpaqueBlack,
  IntOpaqueBlack,
  FloatOpaqueWhite,
  IntOpaqueWhite,
};

enum class SamplerWrapMode {
  Repeat = 0U,
  MirroredRepeat,
  ClampToEdge,
  ClampToBorder,
  MirrorClampToEdge,
};

enum class SamplerCompareOp {
  Never = 0U,
  Less,
  Equal,
  LessOrEqual,
  Greater,
  NotEqual,
  GreaterOrEqual,
  Always,
};

enum class ImageLayout {
  Undefined = 0U,
  General,
  ColorAttachmentOptimal,
  DepthStencilAttachmentOptimal,
  DepthStencilReadOnlyOptimal,
  ShaderReadOnlyOptimal,
  TransferSrcOptimal,
  TransferDstOptimal,
  Preinitialized,
  PresentSrc,
  SharedPresent,
  DepthReadOnlyStencilAttachmentOptimal,
  DepthAttachmentStencilReadOnlyOptimal,
};

enum class PipelineStage {
  TopOfPipe = 0U,
  DrawIndirect,
  VertexInput,
  VertexShader,
  TessellationControlShader,
  TessellationEvaluationShader,
  GeometryShader,
  FragmentShader,
  EarlyFragmentTests,
  LateFragmentTests,
  ColorAttachmentOutput,
  ComputeShader,
  Transfer,
  BottomOfPipe,
  Host,
  AllGraphics,
  AllCommands,
};

enum class AccessFlag {
  Unknown = 0U,
  IndirectCommandRead,
  IndexRead,
  VertexAttributeRead,
  UniformRead,
  InputAttachmentRead,
  ShaderRead,
  ShaderWrite,
  ColorAttachmentRead,
  ColorAttachmentWrite,
  DepthStencilAttachmentRead,
  DepthStencilAttachmentWrite,
  TransferRead,
  TransferWrite,
  HostRead,
  HostWrite,
  MemoryRead,
  MemoryWrite,
};

enum class ShaderStage {
  Vertex = 0U,
  TessellationControl,
  TessellationEvaluation,
  Geometry,
  Fragment,
  Compute,
};

struct ImageSubInfo {
  gpu_ui_connection::GraphicalSize<uint32_t> graphical_size{};
  uint32_t mip_levels = 1U;
  uint32_t array_layers = 1U;
  ImageSampleCount samples{};
  ImageFormat format{};
  ImageDimension dimension{};
};

struct DescriptorInfo {
  vk::ShaderStageFlags stage_flags{};
  uint32_t binding = 0U;
  vk::DescriptorType type{};
  uint32_t size = 0U;
};

struct PushConstantRange {
  vk::ShaderStageFlags stage_flags{};
  uint32_t offset = 0U;
  size_t size = 0U;
};

/// <summary>
/// This struct is not only for image view,
///   but also for image barrier, and etc...
/// It's useful to manage image's miplevel.
/// </summary>
struct ImageViewInfo {
  uint32_t base_mip_level;
  uint32_t mip_levels;
  uint32_t base_array_layer;
  uint32_t array_layers;
  ImageAspect aspect;
};

struct SamplerInfo {
  SamplerFilter mag_filter;
  SamplerFilter min_filter;
  SamplerMipmapMode mipmap_mode;
  SamplerAddressMode address_mode_u;
  SamplerAddressMode address_mode_v;
  SamplerAddressMode address_mode_w;
  float_t mip_lod_bias;
  bool anisotropy_enable;
  float_t max_anisotropy;
  bool compare_enable;
  SamplerCompareOp compare_op;
  float_t min_lod;
  float_t max_lod;
  SamplerBorderColor border_color;
  bool unnormalized_coordinates;
};

namespace gpu {

#ifdef HEPHICS_DEBUG
namespace debug {

/// <summary>
/// This class is a messenger for debugging.
/// This class is used to create a debug instance, and used only in debug mode.
/// </summary>
class Messenger {
 private:
  static std::vector<const char*> s_validationLayers;

  vk::UniqueDebugUtilsMessengerEXT m_ptrMessenger;

 public:
  Messenger() {}
  ~Messenger();

  vk::UniqueInstance createDebugInstance(
      const vk::ApplicationInfo& app_info,
      const std::vector<const char*>& extensions);

  const auto& getValidationLayers() const { return s_validationLayers; }
};

}  // namespace debug
#endif

/// <summary>
/// This class is gpu device wrapper.
/// When gpu resource or memory or etc... are created, this class is used.
/// This class management and operation authority is
///   mainly below hpxc::gpu::Context.
/// </summary>
class Device {
 private:
  vk::PhysicalDevice m_physicalDevice;
  vk::UniqueDevice m_ptrLogicalDevice;

  struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> compute;
    std::optional<uint32_t> transfer;
    std::optional<uint32_t> present;
  } m_queueFamilyIndices;

 public:
  Device(const vk::UniqueInstance& ptr_instance,
         const vk::UniqueSurfaceKHR& ptr_window_surface);
  ~Device();

  const auto& getPhysicalDevice() const { return m_physicalDevice; }
  const auto& getLogicalDevice() const { return m_ptrLogicalDevice; }

  /// <summary>
  /// Search queue family index from queue family type.
  /// Queue in the header file means command queue for gpu's calculation.
  /// Queue family is a group of queues.
  /// Each queue family has a different purpose,
  ///   and also different allowed operations.
  /// </summary>
  /// <param name="family_type">Queue target selection enum</param>
  /// <returns>unsigned 32bit integer: means queue family index</returns>
  const uint32_t getQueueFamilyIndex(const QueueFamilyType family_type) const;

  /// <summary>
  /// Get gpu command queue
  /// </summary>
  /// <param name="queue_family_index">
  /// unsigned 32bit integer:
  ///   '0' or getQueueFamilyIndex member function result
  /// </param>
  /// <returns>vulkan api's queue object</returns>
  vk::Queue getQueue(const uint32_t queue_family_index);

  // vk::Format getSupportedDepthFormat() const;

  void constructLogicalDevice(
#ifdef HEPHICS_DEBUG
      const std::unique_ptr<debug::Messenger>& ptr_messenger
#endif
  );

  /// <summary>
  /// Wait until all gpu operation is done.
  /// From the viewpoint of the performance, this function is not recommended.
  /// This function should be used only application's shutdown.
  /// </summary>
  void waitIdle() const { m_ptrLogicalDevice->waitIdle(); }
};

/// <summary>
/// This class is gpu handler.
/// This class contains
///  vulkan instance, gpu device, window surface, and swapchain.
/// This is the core of Hephics project.
/// </summary>
class Context {
 private:
#ifdef HEPHICS_DEBUG
  std::unique_ptr<debug::Messenger> m_ptrMessenger;
#endif

  vk::UniqueInstance m_ptrInstance;
  std::shared_ptr<gpu_ui_connection::WindowSurface> m_ptrWindowSurface;
  std::unique_ptr<Device> m_ptrDevice;

  bool m_isInitialized = false;

 public:
  Context(std::shared_ptr<gpu_ui_connection::WindowSurface> ptr_window_surface =
              nullptr);
  ~Context();

  const auto& getInstance() const { return m_ptrInstance; }
  const auto& getWindowSurface() const { return m_ptrWindowSurface; }
  const auto& getDevice() const { return m_ptrDevice; }

  bool isInitialized() const { return m_isInitialized; }
};

/// <summary>
/// This class is gpu buffer wrapper.
/// Gpu buffer is used to simple number or matrix
///   about value, vertex, index, uniform, etc ...
/// Also, this class is used to transfer data between cpu and gpu.
/// Buffer size unit is 1byte.
/// So, if you want to float-matrix 4x4,
///   - float size is considered 4bytes - 4 * 4 * 4 = 64bytes are needed.
/// </summary>
class Buffer {
 protected:
  vk::UniqueDeviceMemory m_ptrMemory;
  vk::UniqueBuffer m_ptrBuffer;
  size_t m_size = 0U;

 public:
  Buffer() = default;
  Buffer(const std::unique_ptr<Context>& ptr_context,
         const MemoryUsage memory_usage, const TransferType transfer_type,
         const std::vector<BufferUsage>& buffer_usages, const size_t size);
  ~Buffer();

  Buffer(Buffer&& other) noexcept {
    m_ptrBuffer = std::move(other.m_ptrBuffer);
    m_ptrMemory = std::move(other.m_ptrMemory);
    m_size = other.m_size;
  }

  Buffer& operator=(Buffer&& other) noexcept {
    m_ptrBuffer = std::move(other.m_ptrBuffer);
    m_ptrMemory = std::move(other.m_ptrMemory);
    m_size = other.m_size;
  }

  const auto& getPtrBuffer() const { return m_ptrBuffer; }
  const auto& getBuffer() const { return m_ptrBuffer.get(); }
  auto getSize() const { return m_size; }

  /// <summary>
  /// Get virtual address mapped gpu buffer memory.
  /// Writing or reading data in this address,
  ///   it is directly reflected in gpu memory.
  /// </summary>
  /// <param name="ptr_context"></param>
  /// <returns>virtual gpu buffer memory address</returns>
  void* mapMemory(const std::unique_ptr<Context>& ptr_context) const;

  /// <summary>
  /// Close gpu memory buffer connection.
  /// </summary>
  /// <param name="ptr_context"></param>
  void unmapMemory(const std::unique_ptr<Context>& ptr_context) const;
};

/// <summary>
/// This class is gpu image resource wrapper.
/// </summary>
class Image {
 protected:
  vk::UniqueDeviceMemory m_ptrMemory;
  vk::UniqueImage m_ptrImage;

  uint32_t m_mipLevels = 0U;
  uint32_t m_arrayLayers = 0U;
  vk::Format m_format{};
  ImageDimension m_dimension{};
  gpu_ui_connection::GraphicalSize<uint32_t> m_graphicalSize{};

 public:
  Image() = default;
  Image(const std::unique_ptr<Context>& ptr_context,
        const MemoryUsage memory_usage, const TransferType transfer_type,
        const std::vector<ImageUsage>& image_usages,
        const ImageSubInfo& image_sub_info);
  ~Image();

  Image(Image&& other) noexcept {
    m_ptrMemory = std::move(other.m_ptrMemory);
    m_ptrImage = std::move(other.m_ptrImage);
    m_format = other.m_format;
    m_dimension = other.m_dimension;
    m_graphicalSize = std::move(other.m_graphicalSize);
  }

  Image& operator=(Image&& other) noexcept {
    m_ptrMemory = std::move(other.m_ptrMemory);
    m_ptrImage = std::move(other.m_ptrImage);
    m_format = other.m_format;
    m_dimension = other.m_dimension;
    m_graphicalSize = std::move(other.m_graphicalSize);
  }

  const auto& getPtrImage() const { return m_ptrImage; }
  const auto& getImage() const { return m_ptrImage.get(); }
  auto getMipLevels() const { return m_mipLevels; }
  auto getArrayLayers() const { return m_arrayLayers; }
  auto getFormat() const { return m_format; }
  auto getDimension() const { return m_dimension; }
  const auto& getGraphicalSize() const { return m_graphicalSize; }
};

/// <summary>
/// This class is shader module wrapper.
/// When this class is constructed,
///   shader descriptions are parsed and stored in this class.
/// Spirv-cross shader reflection enabled
///   semi-automation of above shader handling.
/// </summary>
class ShaderModule {
 private:
  vk::UniqueShaderModule m_ptrShaderModule;
  std::string m_entryPointName;

  std::unordered_map<std::string, DescriptorInfo> m_descriptorInfoMap;
  std::unordered_map<std::string, PushConstantRange> m_pushConstantRangeMap;

 public:
  ShaderModule() = default;
  ShaderModule(const std::unique_ptr<Context>& ptr_context,
               const std::vector<uint32_t>& spirv_binary);
  ~ShaderModule();

  ShaderModule(ShaderModule&& other) noexcept {
    m_ptrShaderModule = std::move(other.m_ptrShaderModule);
    m_entryPointName = std::move(other.m_entryPointName);
    m_descriptorInfoMap = std::move(other.m_descriptorInfoMap);
    m_pushConstantRangeMap = std::move(other.m_pushConstantRangeMap);
  };
  ShaderModule& operator=(ShaderModule&& other) noexcept {
    m_ptrShaderModule = std::move(other.m_ptrShaderModule);
    m_entryPointName = std::move(other.m_entryPointName);
    m_descriptorInfoMap = std::move(other.m_descriptorInfoMap);
    m_pushConstantRangeMap = std::move(other.m_pushConstantRangeMap);

    return *this;
  };

  const auto& getModule() const { return m_ptrShaderModule; }
  const auto& getEntryPointName() const { return m_entryPointName; }
  const auto& getDescriptorInfoMap() const { return m_descriptorInfoMap; }
  const auto& getPushConstantRangeMap() const { return m_pushConstantRangeMap; }
};

/// <summary>
/// This class is Hephics-project original idea.
/// For example, in graphics pipeline,
///   uniform-buffer data is often shared on vertex and fragment shaders.
/// However, hpxc::gpu::ShaderModule class is created by only one shader binary.
/// So, this class is used to manage and integrate multiple shader modules.
/// </summary>
class DescriptionUnit {
 private:
  std::unordered_map<std::string, DescriptorInfo> m_descriptorInfoMap;
  std::unordered_map<std::string, PushConstantRange> m_pushConstantRangeMap;

 public:
  DescriptionUnit(
      const std::unordered_map<std::string, ShaderModule>& shader_module_map,
      const std::vector<std::string>& module_keys);
  ~DescriptionUnit();

  const auto& getDescriptorInfoMap() const { return m_descriptorInfoMap; }
  const auto& getPushConstantRangeMap() const { return m_pushConstantRangeMap; }
};

/// <summary>
/// This class is buffer descriptor information
///   for uniform buffer or storage buffer.
/// </summary>
class BufferDescription {
 private:
  std::shared_ptr<vk::WriteDescriptorSet> m_ptrWriteDescriptorSet{};
  std::shared_ptr<vk::DescriptorBufferInfo> m_ptrBufferInfo{};

 public:
  BufferDescription(const DescriptorInfo& descriptor_info,
                    const Buffer& buffer);
  ~BufferDescription();

  const auto& getWriteDescriptorSet() const { return *m_ptrWriteDescriptorSet; }
  const auto& getBufferInfo() const { return *m_ptrBufferInfo; }
};

/// <summary>
/// This class is vulkan image view wrapper.
/// Image view is devided from image resource,
///   because image resource is only memory data and
///   the way to handle image is decided by image view.
/// </summary>
class ImageView {
 private:
  vk::UniqueImageView m_ptrImageView;
  std::unique_ptr<ImageViewInfo> m_ptrImageViewInfo;

 public:
  ImageView(const std::unique_ptr<Context>& ptr_context, const Image& image,
            const ImageViewInfo& image_view_info);
  ~ImageView();

  const auto& getImageView() const { return m_ptrImageView; }
  const auto& getImageViewInfo() const { return *m_ptrImageViewInfo; }
};

/// <summary>
/// This class is vulkan sampler wrapper.
/// Sampler is used to
///   texture filtering, mipmapping,
///   and texture interface on shader.
/// </summary>
class Sampler {
 private:
  vk::UniqueSampler m_ptrSampler;

 public:
  Sampler(const std::unique_ptr<Context>& ptr_context,
          const SamplerInfo& sampler_info);
  ~Sampler();

  const auto& getSampler() const { return m_ptrSampler; }
};

/// <summary>
/// This class is image descriptor information
///   for image, storage image, sampler, and combined image sampler.
/// </summary>
class ImageDescription {
 private:
  std::shared_ptr<vk::WriteDescriptorSet> m_ptrWriteDescriptorSet;
  std::shared_ptr<vk::DescriptorImageInfo> m_ptrImageInfo;

 public:
  /// <summary>
  /// Constructor for only image or storage image
  /// </summary>
  /// <param name="descriptor_info"></param>
  /// <param name="image_view"></param>
  /// <param name="dst_image_layout"></param>
  ImageDescription(const DescriptorInfo& descriptor_info,
                   const ImageView& image_view,
                   const ImageLayout dst_image_layout);
  /// <summary>
  /// Constructor for Sampler
  /// </summary>
  /// <param name="descriptor_info"></param>
  /// <param name="dst_image_layout"></param>
  /// <param name="sampler"></param>
  ImageDescription(const DescriptorInfo& descriptor_info,
                   const ImageLayout dst_image_layout, const Sampler& sampler);
  /// <summary>
  /// Constructor for Combined Image Sampler
  /// </summary>
  /// <param name="descriptor_info"></param>
  /// <param name="image_view"></param>
  /// <param name="dst_image_layout"></param>
  /// <param name="sampler"></param>
  ImageDescription(const DescriptorInfo& descriptor_info,
                   const ImageView& image_view,
                   const ImageLayout dst_image_layout, const Sampler& sampler);
  ~ImageDescription();

  const auto& getWriteDescriptorSet() const { return *m_ptrWriteDescriptorSet; }
  const auto& getImageInfo() const { return *m_ptrImageInfo; }
};

/// <summary>
/// This class is vulkan descriptor set layout wrapper.
/// Descriptor set layout is used to define
///   shader resource binding information.
/// hpxc::gpu::DescriptorSetLayout is devided from hpxc::gpu::DescriptorSet.
/// This class is layout for DescripotrSet,
///   not specific memory resource as DescriptorSet.
/// So, this class is a one-to-many relationship with hpxc::gpu::DescriptorSet.
/// </summary>
class DescriptorSetLayout {
 private:
  vk::UniqueDescriptorSetLayout m_ptrDescriptorSetLayout;
  std::vector<vk::DescriptorPoolSize> m_descriptorPoolSizes;

 public:
  DescriptorSetLayout(const std::unique_ptr<Context>& ptr_context,
                      const DescriptionUnit& description_unit);
  ~DescriptorSetLayout();

  const auto& getDescriptorSetLayout() const {
    return m_ptrDescriptorSetLayout;
  }

  vk::DescriptorPoolCreateInfo getDescriptorPoolInfo() const;
};

/// <summary>
/// This class is vulkan descriptor set wrapper.
/// DescriptorSet is actually used to bind shader resource in gpu.
/// Binding resources are defined by hpxc::gpu::DescriptorSetLayout.
/// </summary>
class DescriptorSet {
 private:
  vk::UniqueDescriptorPool m_ptrDescriptorPool;
  vk::UniqueDescriptorSet m_ptrDescriptorSet;

 public:
  DescriptorSet(const std::unique_ptr<Context>& ptr_context,
                const DescriptorSetLayout& description_set_layout);
  ~DescriptorSet();

  const auto& getDescriptorSet() const { return m_ptrDescriptorSet; }

  /// <summary>
  /// Upload binding resources information to gpu.
  /// Constructing this class, it is allocation of gpu memory
  ///   for shader binding data.
  /// Thus, this funciont is used to register descriptions.
  /// Of cource,
  ///   hpxc::gpu::DescriptorSetLayout object's definition must be followed.
  /// </summary>
  /// <param name="ptr_context"></param>
  /// <param name="buffer_descriptions"></param>
  /// <param name="image_descriptions"></param>
  void updateDescriptorSet(
      const std::unique_ptr<Context>& ptr_context,
      const std::vector<BufferDescription>& buffer_descriptions,
      const std::vector<ImageDescription>& image_descriptions);

  /// <summary>
  /// Free gpu memory for descriptor set
  ///   for next updateDescriptorSet function call.
  /// This function usage:
  ///   if you want to change image resource data
  ///     without changing shader module or description unit,
  ///   this function is very useful and ecological for gpu memory.
  /// </summary>
  /// <param name="ptr_context"></param>
  void freeDescriptorSet(const std::unique_ptr<Context>& ptr_context);
};

/// <summary>
/// This class is vulkan pipeline and pipeline layout wrapper.
/// In Hephics project,
///   considered that dividation of pipeline and its layout is not needed.
/// </summary>
class Pipeline {
 private:
  vk::UniquePipeline m_ptrPipeline;
  vk::UniquePipelineLayout m_ptrPipelineLayout;
  QueueFamilyType m_queueFamilyType{};

 public:
  Pipeline(const std::unique_ptr<Context>& ptr_context,
           const DescriptionUnit& description_unit,
           const DescriptorSetLayout& descriptor_set_layout);
  ~Pipeline();

  const auto& getPipeline() const { return m_ptrPipeline; }
  const auto& getPipelineLayout() const { return m_ptrPipelineLayout; }
  const auto getQueueFamilyType() const { return m_queueFamilyType; }

  /// <summary>
  /// Construt pipeline for compute shader.
  /// </summary>
  /// <param name="ptr_context"></param>
  /// <param name="shader_module"></param>
  void constructComputePipeline(const std::unique_ptr<Context>& ptr_context,
                                const ShaderModule& shader_module);
};

/// <summary>
/// This class is vulkan buffer memory barrier wrapper.
/// This class is used to synchronize buffer memory access.
/// And, this class is also used to change buffer owner
///   from one queue to another
///   - in case, set the appropriate pair of queue family index: src, dst
///   - and, two barriers are needed: release, acquire.
/// </summary>
class BufferBarrier {
 private:
  vk::BufferMemoryBarrier m_bufferMemoryBarrier{};

 public:
  BufferBarrier(const Buffer& buffer,
                const std::vector<AccessFlag>& priority_access_flags,
                const std::vector<AccessFlag> wait_access_flags);
  ~BufferBarrier();

  auto& getBarrier() { return m_bufferMemoryBarrier; }
  const auto& getBarrier() const { return m_bufferMemoryBarrier; }

  void setSrcQueueFamilyIndex(const uint32_t index) {
    m_bufferMemoryBarrier.setSrcQueueFamilyIndex(index);
  }

  void setDstQueueFamilyIndex(const uint32_t index) {
    m_bufferMemoryBarrier.setDstQueueFamilyIndex(index);
  }
};

/// <summary>
/// This class is vulkan image memory barrier wrapper.
/// This class is used to synchronize image memory access.
/// And, this class is also used to change image layout.
/// This class is also used to change buffer owner
///   from one queue to another
///   - in case, set the appropriate pair of queue family index: src, dst
///   - and, two barriers are needed: release, acquire.
/// Notice: when you want to change owner of image,
///   don't change image layout in release barrier, but in acquire barrier.
/// </summary>
class ImageBarrier {
 private:
  vk::ImageMemoryBarrier m_imageMemoryBarrier{};

 public:
  ImageBarrier(const Image& image,
               const std::vector<AccessFlag>& priority_access_flags,
               const std::vector<AccessFlag> wait_access_flags,
               const ImageLayout old_layout, const ImageLayout new_layout,
               const ImageViewInfo& image_view_info);
  ~ImageBarrier();

  auto& getBarrier() { return m_imageMemoryBarrier; }
  const auto& getBarrier() const { return m_imageMemoryBarrier; }

  void setSrcQueueFamilyIndex(const uint32_t index) {
    m_imageMemoryBarrier.setSrcQueueFamilyIndex(index);
  }

  void setDstQueueFamilyIndex(const uint32_t index) {
    m_imageMemoryBarrier.setDstQueueFamilyIndex(index);
  }
};

/// <summary>
/// This class is vulkan timeline semaphore wrapper
///   (Vulkan API version must be more than 1.2).
/// This class is used to synchronize cpu and gpu operation.
/// </summary>
class Semaphore {
 private:
  vk::UniqueSemaphore m_ptrSemaphore;
  uint64_t m_signalValue = 1U;
  uint64_t m_waitValue = 0U;
  vk::TimelineSemaphoreSubmitInfoKHR m_timelineSubmitInfo{};
  std::vector<vk::PipelineStageFlags> m_waitStages{};

 public:
  Semaphore(const std::unique_ptr<Context>& ptr_context);
  ~Semaphore();

  const auto& getSemaphore() const { return m_ptrSemaphore; }
  const auto& getSignalValue() const { return m_signalValue; }
  const auto& getWaitValue() const { return m_waitValue; }
  auto getPtrTimelineSubmitInfo() const { return &m_timelineSubmitInfo; }
  const auto& getBackWaitStage() const { return m_waitStages.back(); }
  const auto& getWaitStage() const { return m_waitStages; }

  void setWaitStage(const vk::PipelineStageFlagBits wait_stage) {
    m_waitStages.push_back(wait_stage);
  }

  void updateSignalValue() { m_signalValue += 1; }
  void updateWaitValue() { m_waitValue += 1; }

  /// <summary>
  /// Wait until finishing gpu operation submitted with this semaphore.
  /// </summary>
  /// <param name="ptr_context"></param>
  void wait(const std::unique_ptr<Context>& ptr_context);
};

}  // namespace gpu
}  // namespace hpxc
