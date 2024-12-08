#include "computing_frames_handle.h"

static uint32_t calculate_mip_levels(
    const hpxc::gpu_ui_connection::GraphicalSize<uint32_t>& size) {
  return static_cast<uint32_t>(
      std::floor(std::log2(std::max(size.width, size.height))) + 1U);
}

samples::core::ComputingFramesHandle::ComputingFramesHandle() {
  m_ptrContext = std::make_unique<hpxc::gpu::Context>(nullptr);

  m_ptrComputeCommandDriver.reset(
      new hpxc::CommandDriver(m_ptrContext, hpxc::QueueFamilyType::Compute));
  m_ptrTransferCommandDriver.reset(
      new hpxc::CommandDriver(m_ptrContext, hpxc::QueueFamilyType::Transfer));
  m_ptrUniformBuffer.reset(
      hpxc::createPtrUniformBuffer(m_ptrContext, sizeof(float_t)));
  const auto mapped_address = m_ptrUniformBuffer->mapMemory(m_ptrContext);
  std::fill_n(reinterpret_cast<float_t*>(mapped_address),
              m_ptrUniformBuffer->getSize() / sizeof(float_t), 5.0f);
  m_ptrUniformBuffer->unmapMemory(m_ptrContext);

  initializeImageResources();
  constructShaderResources();
}

samples::core::ComputingFramesHandle::~ComputingFramesHandle() {
  m_ptrContext->getDevice()->waitIdle();
}

void samples::core::ComputingFramesHandle::run() {
  hpxc::gpu::Semaphore semaphore(m_ptrContext);

  // prepare resources for compute
  {
    std::vector<hpxc::gpu::Buffer> staging_buffers;
    setResourceTransferCommands(staging_buffers);
    setResourceReceiveCommands();

    m_ptrTransferCommandDriver->submit(hpxc::PipelineStage::BottomOfPipe,
                                       semaphore);
    m_ptrComputeCommandDriver->submit(hpxc::PipelineStage::Transfer, semaphore);
    semaphore.wait(m_ptrContext);

    m_ptrTransferCommandDriver->resetAllCommandPools(m_ptrContext);
    m_ptrComputeCommandDriver->resetAllCommandPools(m_ptrContext);
  }

  // computing loop
  while (true) {
    const auto result_buffer = hpxc::createStagingBufferFromGPU(
        m_ptrContext, m_image.total() * m_image.elemSize());
    setComputeCommands(result_buffer);

    m_ptrComputeCommandDriver->submit(hpxc::PipelineStage::ComputeShader,
                                      semaphore);
    semaphore.wait(m_ptrContext);

    const auto result_mapped_address = result_buffer.mapMemory(m_ptrContext);
    const auto& image_size = m_ptrStorageImage->getGraphicalSize();
    cv::Mat result(image_size.height, image_size.width, CV_8UC4,
                   result_mapped_address);
    result_buffer.unmapMemory(m_ptrContext);
    cv::cvtColor(result, result, cv::COLOR_RGBA2BGR);

    cv::imshow("Result", result);
    const auto cv_input = cv::waitKey(1);
    if (cv_input == static_cast<decltype(cv_input)>('q')) {
      cv::destroyWindow("Result");
      break;
    }

    m_ptrComputeCommandDriver->resetAllCommandPools(m_ptrContext);
  }
}

void samples::core::ComputingFramesHandle::initializeImageResources() {
  m_image = cv::imread("images/lenna.png");
  cv::cvtColor(m_image, m_image, cv::COLOR_BGR2RGBA);

  hpxc::ImageSubInfo image_sub_info;
  image_sub_info.graphical_size.width =
      static_cast<uint32_t>(m_image.size().width);
  image_sub_info.graphical_size.height =
      static_cast<uint32_t>(m_image.size().height);
  image_sub_info.graphical_size.depth = 1U;
  image_sub_info.mip_levels =
      calculate_mip_levels(image_sub_info.graphical_size);
  image_sub_info.array_layers = 1U;
  image_sub_info.samples = hpxc::ImageSampleCount::v1;
  image_sub_info.format = hpxc::ImageFormat::R8G8B8A8Unorm;
  image_sub_info.dimension = hpxc::ImageDimension::v2D;

  m_ptrImage.reset(
      new hpxc::gpu::Image(m_ptrContext, hpxc::MemoryUsage::GpuOnly,
                           hpxc::TransferType::TransferSrcDst,
                           {hpxc::ImageUsage::Sampled}, image_sub_info));

  m_ptrStorageImage.reset(
      new hpxc::gpu::Image(m_ptrContext, hpxc::MemoryUsage::GpuOnly,
                           hpxc::TransferType::TransferSrcDst,
                           {hpxc::ImageUsage::Storage}, image_sub_info));

  {
    hpxc::ImageViewInfo image_view_info{};
    image_view_info.aspect = hpxc::ImageAspect::Color;
    image_view_info.base_array_layer = 0U;
    image_view_info.array_layers = image_sub_info.array_layers;
    image_view_info.base_mip_level = 0U;
    image_view_info.mip_levels = image_sub_info.mip_levels;

    m_ptrImageView.reset(
        new hpxc::gpu::ImageView(m_ptrContext, *m_ptrImage, image_view_info));

    m_ptrStorageImageView.reset(new hpxc::gpu::ImageView(
        m_ptrContext, *m_ptrStorageImage, image_view_info));
  }

  {
    hpxc::SamplerInfo sampler_info{};
    sampler_info.address_mode_u = hpxc::SamplerAddressMode::ClampToBorder;
    sampler_info.address_mode_v = hpxc::SamplerAddressMode::ClampToBorder;
    sampler_info.address_mode_w = hpxc::SamplerAddressMode::ClampToBorder;
    sampler_info.mag_filter = hpxc::SamplerFilter::Linear;
    sampler_info.min_filter = hpxc::SamplerFilter::Linear;
    sampler_info.mipmap_mode = hpxc::SamplerMipmapMode::Linear;
    sampler_info.mip_lod_bias = 0.0f;
    sampler_info.anisotropy_enable = false;
    sampler_info.compare_enable = false;
    sampler_info.max_lod = static_cast<float_t>(image_sub_info.mip_levels);
    sampler_info.min_lod = 0.0f;
    sampler_info.border_color = hpxc::SamplerBorderColor::FloatOpaqueWhite;
    sampler_info.unnormalized_coordinates = false;

    m_ptrImageSampler.reset(new hpxc::gpu::Sampler(m_ptrContext, sampler_info));
  }
}

void samples::core::ComputingFramesHandle::constructShaderResources() {
  const auto spirv_binary =
      hpxc::io::shader::read("shaders/compute/simple_image.comp");

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

  std::vector<hpxc::gpu::ImageDescription> image_descriptions;
  image_descriptions.emplace_back(
      description_unit.getDescriptorInfoMap().at("image"), *m_ptrImageView,
      hpxc::ImageLayout::ShaderReadOnlyOptimal, *m_ptrImageSampler);
  image_descriptions.emplace_back(
      description_unit.getDescriptorInfoMap().at("dest_image"),
      *m_ptrStorageImageView, hpxc::ImageLayout::General);

  m_ptrDescriptorSet->updateDescriptorSet(m_ptrContext, buffer_descriptions,
                                          image_descriptions);

  m_ptrComputePipeline.reset(new hpxc::gpu::Pipeline(
      m_ptrContext, description_unit, *m_ptrDescriptorSetLayout));
  m_ptrComputePipeline->constructComputePipeline(
      m_ptrContext, m_shaderModuleMap.at("compute"));
}

void samples::core::ComputingFramesHandle::setResourceTransferCommands(
    std::vector<hpxc::gpu::Buffer>& staging_buffers) {
  const auto command_buffer = m_ptrTransferCommandDriver->getTransfer();

  staging_buffers.push_back(hpxc::createStagingBufferToGPU(
      m_ptrContext, m_image.total() * m_image.elemSize()));

  auto& staging_buffer = staging_buffers.back();
  const auto mapped_address = staging_buffer.mapMemory(m_ptrContext);
  std::memcpy(mapped_address, m_image.data, staging_buffer.getSize());
  staging_buffer.unmapMemory(m_ptrContext);

  command_buffer.begin();

  const hpxc::ImageViewInfo image_view_info =
      m_ptrImageView->getImageViewInfo();

  const auto image_barrier = hpxc::gpu::ImageBarrier(
      *m_ptrImage, {hpxc::AccessFlag::Unknown},
      {hpxc::AccessFlag::TransferWrite}, hpxc::ImageLayout::Undefined,
      hpxc::ImageLayout::TransferDstOptimal, image_view_info);
  command_buffer.setPipelineBarrier(image_barrier,
                                    hpxc::PipelineStage::TopOfPipe,
                                    hpxc::PipelineStage::Transfer);

  command_buffer.copyBufferToImage(staging_buffer, *m_ptrImage,
                                   hpxc::ImageLayout::TransferDstOptimal,
                                   image_view_info);

  // mipmap generation and release the ownership
  command_buffer.setMipmaps(*m_ptrImage, hpxc::PipelineStage::Transfer);

  command_buffer.transferMipmapImages(
      *m_ptrImage, hpxc::PipelineStage::Transfer,
      hpxc::PipelineStage::BottomOfPipe,
      {m_ptrTransferCommandDriver->getQueueFamilyIndex(),
       m_ptrComputeCommandDriver->getQueueFamilyIndex()});

  command_buffer.end();
}

void samples::core::ComputingFramesHandle::setResourceReceiveCommands() {
  const auto command_buffer = m_ptrComputeCommandDriver->getCompute();
  command_buffer.begin();

  command_buffer.acquireMipmapImages(
      *m_ptrImage, hpxc::PipelineStage::BottomOfPipe,
      hpxc::PipelineStage::ComputeShader,
      {m_ptrTransferCommandDriver->getQueueFamilyIndex(),
       m_ptrComputeCommandDriver->getQueueFamilyIndex()});

  const hpxc::ImageViewInfo image_view_info =
      m_ptrStorageImageView->getImageViewInfo();

  const auto image_barrier = hpxc::gpu::ImageBarrier(
      *m_ptrStorageImage, {hpxc::AccessFlag::Unknown},
      {hpxc::AccessFlag::ShaderWrite}, hpxc::ImageLayout::Undefined,
      hpxc::ImageLayout::General, image_view_info);

  command_buffer.setPipelineBarrier(image_barrier,
                                    hpxc::PipelineStage::TopOfPipe,
                                    hpxc::PipelineStage::ComputeShader);

  command_buffer.end();
}

void samples::core::ComputingFramesHandle::setComputeCommands(
    const hpxc::gpu::Buffer& staging_buffer) {
  static float_t push_timer = 0.0f;
  push_timer += 0.001f;

  const auto command_buffer = m_ptrComputeCommandDriver->getCompute();
  command_buffer.begin();

  command_buffer.pushConstants(*m_ptrComputePipeline,
                               {hpxc::ShaderStage::Compute}, 0U, {push_timer});
  command_buffer.compute(*m_ptrComputePipeline, *m_ptrDescriptorSet,
                         hpxc::ComputeWorkGroupSize{
                             m_ptrImage->getGraphicalSize().width / 4U,
                             m_ptrImage->getGraphicalSize().height / 4U, 1U});

  const hpxc::ImageViewInfo image_view_info =
      m_ptrStorageImageView->getImageViewInfo();

  {
    const auto image_barrier = hpxc::gpu::ImageBarrier(
        *m_ptrStorageImage, {hpxc::AccessFlag::ShaderWrite},
        {hpxc::AccessFlag::TransferRead}, hpxc::ImageLayout::General,
        hpxc::ImageLayout::General, image_view_info);

    command_buffer.setPipelineBarrier(image_barrier,
                                      hpxc::PipelineStage::ComputeShader,
                                      hpxc::PipelineStage::Transfer);
  }

  command_buffer.copyImageToBuffer(*m_ptrStorageImage, staging_buffer,
                                   hpxc::ImageLayout::General, image_view_info);

  // set barrier for next loop shader writing
  {
    const auto image_barrier = hpxc::gpu::ImageBarrier(
        *m_ptrStorageImage, {hpxc::AccessFlag::TransferRead},
        {hpxc::AccessFlag::ShaderWrite}, hpxc::ImageLayout::General,
        hpxc::ImageLayout::General, image_view_info);
    command_buffer.setPipelineBarrier(image_barrier,
                                      hpxc::PipelineStage::Transfer,
                                      hpxc::PipelineStage::ComputeShader);
  }

  command_buffer.end();
}
