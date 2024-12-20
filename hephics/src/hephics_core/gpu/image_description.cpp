#include "../gpu.hpp"
#include "vk_helper.hpp"

hpxc::gpu::ImageDescription::ImageDescription(
    const DescriptorInfo& descriptor_info, const ImageView& image_view,
    const ImageLayout dst_image_layout) {
  m_ptrImageInfo = std::make_shared<vk::DescriptorImageInfo>();
  m_ptrWriteDescriptorSet = std::make_shared<vk::WriteDescriptorSet>();

  m_ptrImageInfo->setImageLayout(vk_helper::getImageLayout(dst_image_layout));
  m_ptrImageInfo->setImageView(image_view.getImageView().get());

  m_ptrWriteDescriptorSet->setDstBinding(descriptor_info.binding);
  m_ptrWriteDescriptorSet->setDstArrayElement(0U);
  m_ptrWriteDescriptorSet->setDescriptorType(descriptor_info.type);
  m_ptrWriteDescriptorSet->setImageInfo(*m_ptrImageInfo);
}

hpxc::gpu::ImageDescription::ImageDescription(
    const DescriptorInfo& descriptor_info, const ImageLayout dst_image_layout,
    const Sampler& sampler) {
  m_ptrImageInfo = std::make_shared<vk::DescriptorImageInfo>();
  m_ptrWriteDescriptorSet = std::make_shared<vk::WriteDescriptorSet>();

  m_ptrImageInfo->setImageLayout(vk_helper::getImageLayout(dst_image_layout));
  m_ptrImageInfo->setSampler(sampler.getSampler().get());

  m_ptrWriteDescriptorSet->setDstBinding(descriptor_info.binding);
  m_ptrWriteDescriptorSet->setDstArrayElement(0U);
  m_ptrWriteDescriptorSet->setDescriptorType(descriptor_info.type);
  m_ptrWriteDescriptorSet->setImageInfo(*m_ptrImageInfo);
}

hpxc::gpu::ImageDescription::ImageDescription(
    const DescriptorInfo& descriptor_info, const ImageView& image_view,
    const ImageLayout dst_image_layout, const Sampler& sampler) {
  m_ptrImageInfo = std::make_shared<vk::DescriptorImageInfo>();
  m_ptrWriteDescriptorSet = std::make_shared<vk::WriteDescriptorSet>();

  m_ptrImageInfo->setImageLayout(vk_helper::getImageLayout(dst_image_layout));
  m_ptrImageInfo->setImageView(image_view.getImageView().get());
  m_ptrImageInfo->setSampler(sampler.getSampler().get());

  m_ptrWriteDescriptorSet->setDstBinding(descriptor_info.binding);
  m_ptrWriteDescriptorSet->setDstArrayElement(0U);
  m_ptrWriteDescriptorSet->setDescriptorType(descriptor_info.type);
  m_ptrWriteDescriptorSet->setImageInfo(*m_ptrImageInfo);
}

hpxc::gpu::ImageDescription::~ImageDescription() {}
