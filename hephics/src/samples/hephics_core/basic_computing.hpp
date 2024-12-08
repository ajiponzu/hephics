#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../../hephics_core.hpp"

namespace samples {
namespace core {

class BasicComputing {
 private:
  std::unique_ptr<hpxc::gpu::Context> m_ptrContext;

  std::unique_ptr<hpxc::CommandDriver> m_ptrComputeCommandDriver;
  std::unique_ptr<hpxc::CommandDriver> m_ptrTransferCommandDriver;

  std::unique_ptr<hpxc::gpu::Buffer> m_ptrUniformBuffer;
  std::unique_ptr<hpxc::gpu::Buffer> m_ptrInputStorageBuffer;
  std::unique_ptr<hpxc::gpu::Buffer> m_ptrOutputStorageBuffer;

  hpxc::ShaderModuleMap m_shaderModuleMap;

  std::unique_ptr<hpxc::gpu::DescriptorSetLayout> m_ptrDescriptorSetLayout;

  std::unique_ptr<hpxc::gpu::DescriptorSet> m_ptrDescriptorSet;
  std::unique_ptr<hpxc::gpu::Pipeline> m_ptrComputePipeline;

 public:
  BasicComputing();

  ~BasicComputing();

  void run();

 private:
  void setTransferCommands(std::vector<hpxc::gpu::Buffer>& staging_buffers);

  void constructShaderResources();
  void setComputeCommands(hpxc::gpu::Buffer& staging_buffer);
};

}  // namespace core
}  // namespace samples
