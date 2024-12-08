#include "../gpu.hpp"
#include "vk_helper.hpp"

static vk::ImageUsageFlags get_transfer_usage(
    const hpxc::TransferType transfer_type) {
  using TransferType = hpxc::TransferType;

  switch (transfer_type) {
    case TransferType::TransferSrc:
      return vk::ImageUsageFlagBits::eTransferSrc;
    case TransferType::TransferDst:
      return vk::ImageUsageFlagBits::eTransferDst;
    case TransferType::TransferSrcDst:
      return vk::ImageUsageFlagBits::eTransferSrc |
             vk::ImageUsageFlagBits::eTransferDst;
    default:
      return vk::ImageUsageFlagBits::eTransferSrc;
  }
}

static vk::ImageUsageFlagBits get_image_usage(
    const hpxc::ImageUsage image_usage) {
  using ImageUsage = hpxc::ImageUsage;

  switch (image_usage) {
    case ImageUsage::Sampled:
      return vk::ImageUsageFlagBits::eSampled;
    case ImageUsage::Storage:
      return vk::ImageUsageFlagBits::eStorage;
    case ImageUsage::ColorAttachment:
      return vk::ImageUsageFlagBits::eColorAttachment;
    case ImageUsage::DepthStencilAttachment:
      return vk::ImageUsageFlagBits::eDepthStencilAttachment;
    case ImageUsage::TransientAttachment:
      return vk::ImageUsageFlagBits::eTransientAttachment;
    case ImageUsage::InputAttachment:
      return vk::ImageUsageFlagBits::eInputAttachment;
    default:
      return vk::ImageUsageFlagBits::eSampled;
  }
}

static vk::ImageType get_image_type(hpxc::ImageDimension dimension) {
  using ImageDimension = hpxc::ImageDimension;

  switch (dimension) {
    case ImageDimension::v1D:
      return vk::ImageType::e1D;
    case ImageDimension::v2D:
      return vk::ImageType::e2D;
    case ImageDimension::v3D:
      return vk::ImageType::e3D;
    default:
      return vk::ImageType::e2D;
  }
}

static vk::SampleCountFlagBits get_sample_Count(
    const hpxc::ImageSampleCount sample_count) {
  using ImageSampleCount = hpxc::ImageSampleCount;

  switch (sample_count) {
    case ImageSampleCount::v1:
      return vk::SampleCountFlagBits::e1;
    case ImageSampleCount::v2:
      return vk::SampleCountFlagBits::e2;
    case ImageSampleCount::v4:
      return vk::SampleCountFlagBits::e4;
    case ImageSampleCount::v8:
      return vk::SampleCountFlagBits::e8;
    case ImageSampleCount::v16:
      return vk::SampleCountFlagBits::e16;
    case ImageSampleCount::v32:
      return vk::SampleCountFlagBits::e32;
    case ImageSampleCount::v64:
      return vk::SampleCountFlagBits::e64;
    default:
      return vk::SampleCountFlagBits::e1;
  }
}

hpxc::gpu::Image::Image(const std::unique_ptr<Context>& ptr_context,
                        const MemoryUsage memory_usage,
                        const TransferType transfer_type,
                        const std::vector<ImageUsage>& image_usages,
                        const ImageSubInfo& image_sub_info) {
  {
    vk::ImageCreateInfo image_info{};
    {
      const vk::ImageUsageFlags vk_transfer_type =
          get_transfer_usage(transfer_type);

      vk::ImageUsageFlags vk_image_usages{};
      for (const auto& image_usage : image_usages) {
        vk_image_usages |= get_image_usage(image_usage);
      }

      image_info.setUsage(vk_transfer_type | vk_image_usages);
    }

    {
      const vk::Format vk_format =
          vk_helper::getImageFormat(image_sub_info.format);

      image_info.setFormat(vk_format);
      m_format = vk_format;
    }

    {
      vk::Extent3D vk_extent{};
      vk_extent.setWidth(image_sub_info.graphical_size.width);
      vk_extent.setHeight(image_sub_info.graphical_size.height);
      vk_extent.setDepth(image_sub_info.graphical_size.depth);

      image_info.setExtent(vk_extent);
      m_graphicalSize = image_sub_info.graphical_size;
    }

    m_arrayLayers = image_sub_info.array_layers;
    image_info.setArrayLayers(m_arrayLayers);

    m_mipLevels = image_sub_info.mip_levels;
    image_info.setMipLevels(image_sub_info.mip_levels);

    image_info.setImageType(get_image_type(image_sub_info.dimension));
    m_dimension = image_sub_info.dimension;

    image_info.setSamples(get_sample_Count(image_sub_info.samples));
    image_info.setTiling(vk::ImageTiling::eOptimal);
    image_info.setSharingMode(vk::SharingMode::eExclusive);
    image_info.setInitialLayout(vk::ImageLayout::eUndefined);
    image_info.setQueueFamilyIndexCount(0U);
    image_info.setPQueueFamilyIndices(nullptr);

    m_ptrImage =
        ptr_context->getDevice()->getLogicalDevice()->createImageUnique(
            image_info);
  }

  {
    const auto memory_requirements =
        ptr_context->getDevice()
            ->getLogicalDevice()
            ->getImageMemoryRequirements(m_ptrImage.get());

    const vk::MemoryPropertyFlags vk_memory_usage =
        vk_helper::getMemoryPropertyFlags(memory_usage);

    const auto memory_props =
        ptr_context->getDevice()->getPhysicalDevice().getMemoryProperties();

    uint32_t memory_type_idx = 0U;
    for (; memory_type_idx < memory_props.memoryTypeCount;
         memory_type_idx += 1U) {
      if ((memory_requirements.memoryTypeBits & (1U << memory_type_idx)) &&
          (memory_props.memoryTypes.at(memory_type_idx).propertyFlags &
           vk_memory_usage) == vk_memory_usage) {
        break;
      }
    }

    vk::MemoryAllocateInfo allocation_info{};
    allocation_info.setMemoryTypeIndex(memory_type_idx);
    allocation_info.setAllocationSize(memory_requirements.size);

    m_ptrMemory =
        ptr_context->getDevice()->getLogicalDevice()->allocateMemoryUnique(
            allocation_info);
  }

  ptr_context->getDevice()->getLogicalDevice()->bindImageMemory(
      m_ptrImage.get(), m_ptrMemory.get(), 0U);
}

hpxc::gpu::Image::~Image() {}
