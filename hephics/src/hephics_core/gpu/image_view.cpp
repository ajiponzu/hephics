#include "../gpu.hpp"
#include "vk_helper.hpp"

static vk::ImageViewType get_image_view_type(
    const hpxc::ImageDimension image_dimension) {
  switch (image_dimension) {
    case hpxc::ImageDimension::v1D:
      return vk::ImageViewType::e1D;
    case hpxc::ImageDimension::v2D:
      return vk::ImageViewType::e2D;
    case hpxc::ImageDimension::v3D:
      return vk::ImageViewType::e3D;
    default:
      return vk::ImageViewType::e2D;
  }
}

hpxc::gpu::ImageView::ImageView(
    const std::unique_ptr<Context>& ptr_context, const Image& image,
    const ImageViewInfo& image_view_info) {
  vk::ImageViewCreateInfo create_info;
  {
    vk::ImageSubresourceRange subresource_range;
    subresource_range.setAspectMask(
        vk_helper::getImageAspectFlags(image_view_info.aspect));
    subresource_range.setBaseMipLevel(image_view_info.base_mip_level);
    subresource_range.setLevelCount(image_view_info.mip_levels);
    subresource_range.setBaseArrayLayer(image_view_info.base_array_layer);
    subresource_range.setLayerCount(image_view_info.array_layers);

    create_info.setSubresourceRange(subresource_range);
  }

  {
    vk::ComponentMapping component_mapping;
    component_mapping.setR(vk::ComponentSwizzle::eIdentity);
    component_mapping.setG(vk::ComponentSwizzle::eIdentity);
    component_mapping.setB(vk::ComponentSwizzle::eIdentity);
    component_mapping.setA(vk::ComponentSwizzle::eIdentity);

    create_info.setComponents(component_mapping);
  }

  create_info.setViewType(get_image_view_type(image.getDimension()));
  create_info.setFormat(image.getFormat());
  create_info.setImage(image.getImage());

  m_ptrImageView =
      ptr_context->getDevice()->getLogicalDevice()->createImageViewUnique(
          create_info);
  m_ptrImageViewInfo.reset(new ImageViewInfo(image_view_info));
}

hpxc::gpu::ImageView::~ImageView() {}
