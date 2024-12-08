#pragma once

#include <opencv2/opencv.hpp>

#include "../../hephics_core.hpp"

namespace samples {

namespace core {

class SimpleImageComputing {
 private:
  std::unique_ptr<hpxc::gpu::Context> m_ptrContext;

  std::unique_ptr<hpxc::gpu::Image> m_ptrImage;
  std::unique_ptr<hpxc::gpu::Image> m_ptrStorageImage;
  std::unique_ptr<hpxc::gpu::Buffer> m_ptrUniformBuffer;

  std::unique_ptr<hpxc::gpu::ImageView> m_ptrImageView;
  std::unique_ptr<hpxc::gpu::ImageView> m_ptrStorageImageView;

  std::unique_ptr<hpxc::gpu::Sampler> m_ptrImageSampler;

  std::unique_ptr<hpxc::CommandDriver> m_ptrComputeCommandDriver;
  std::unique_ptr<hpxc::CommandDriver> m_ptrTransferCommandDriver;

  hpxc::ShaderModuleMap m_shaderModuleMap;

  std::unique_ptr<hpxc::gpu::DescriptorSetLayout> m_ptrDescriptorSetLayout;

  std::unique_ptr<hpxc::gpu::DescriptorSet> m_ptrDescriptorSet;
  std::unique_ptr<hpxc::gpu::Pipeline> m_ptrComputePipeline;

  cv::Mat m_image;

 public:
  SimpleImageComputing();
  ~SimpleImageComputing();

  void run();

 private:
  void initializeImageResources();
  void constructShaderResources();

  void setTransferCommands(std::vector<hpxc::gpu::Buffer>& staging_buffers);

  void setComputeCommands(const hpxc::gpu::Buffer& staging_buffer);
};

}  // namespace core
}  // namespace samples
