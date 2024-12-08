#include "../gpu.hpp"

hpxc::gpu::Sampler::Sampler(const std::unique_ptr<Context>& ptr_context,
                            const SamplerInfo& sampler_info) {
  vk::SamplerCreateInfo sampler_create_info;
  {
    switch (sampler_info.mag_filter) {
      case SamplerFilter::Linear:
        sampler_create_info.setMagFilter(vk::Filter::eLinear);
        break;
      case SamplerFilter::Nearest:
        sampler_create_info.setMagFilter(vk::Filter::eNearest);
        break;
      case SamplerFilter::Cubic:
        sampler_create_info.setMagFilter(vk::Filter::eCubicIMG);
        break;
      default:
        break;
    }

    switch (sampler_info.min_filter) {
      case SamplerFilter::Linear:
        sampler_create_info.setMinFilter(vk::Filter::eLinear);
        break;
      case SamplerFilter::Nearest:
        sampler_create_info.setMinFilter(vk::Filter::eNearest);
        break;
      case SamplerFilter::Cubic:
        sampler_create_info.setMinFilter(vk::Filter::eCubicIMG);
        break;
      default:
        break;
    }

    switch (sampler_info.mipmap_mode) {
      case SamplerMipmapMode::Linear:
        sampler_create_info.setMipmapMode(vk::SamplerMipmapMode::eLinear);
        break;
      case SamplerMipmapMode::Nearest:
        sampler_create_info.setMipmapMode(vk::SamplerMipmapMode::eNearest);
        break;
      default:
        break;
    }

    switch (sampler_info.address_mode_u) {
      case SamplerAddressMode::Repeat:
        sampler_create_info.setAddressModeU(vk::SamplerAddressMode::eRepeat);
        break;
      case SamplerAddressMode::MirroredRepeat:
        sampler_create_info.setAddressModeU(
            vk::SamplerAddressMode::eMirroredRepeat);
        break;
      case SamplerAddressMode::ClampToEdge:
        sampler_create_info.setAddressModeU(
            vk::SamplerAddressMode::eClampToEdge);
        break;
      case SamplerAddressMode::ClampToBorder:
        sampler_create_info.setAddressModeU(
            vk::SamplerAddressMode::eClampToBorder);
        break;
      case SamplerAddressMode::MirrorClampToEdge:
        sampler_create_info.setAddressModeU(
            vk::SamplerAddressMode::eMirrorClampToEdge);
        break;
      default:
        break;
    }

    switch (sampler_info.address_mode_v) {
      case SamplerAddressMode::Repeat:
        sampler_create_info.setAddressModeV(vk::SamplerAddressMode::eRepeat);
        break;
      case SamplerAddressMode::MirroredRepeat:
        sampler_create_info.setAddressModeV(
            vk::SamplerAddressMode::eMirroredRepeat);
        break;
      case SamplerAddressMode::ClampToEdge:
        sampler_create_info.setAddressModeV(
            vk::SamplerAddressMode::eClampToEdge);
        break;
      case SamplerAddressMode::ClampToBorder:
        sampler_create_info.setAddressModeV(
            vk::SamplerAddressMode::eClampToBorder);
        break;
      case SamplerAddressMode::MirrorClampToEdge:
        sampler_create_info.setAddressModeV(
            vk::SamplerAddressMode::eMirrorClampToEdge);
        break;
      default:
        break;
    }

    switch (sampler_info.address_mode_w) {
      case SamplerAddressMode::Repeat:
        sampler_create_info.setAddressModeW(vk::SamplerAddressMode::eRepeat);
        break;
      case SamplerAddressMode::MirroredRepeat:
        sampler_create_info.setAddressModeW(
            vk::SamplerAddressMode::eMirroredRepeat);
        break;
      case SamplerAddressMode::ClampToEdge:
        sampler_create_info.setAddressModeW(
            vk::SamplerAddressMode::eClampToEdge);
        break;
      case SamplerAddressMode::ClampToBorder:
        sampler_create_info.setAddressModeW(
            vk::SamplerAddressMode::eClampToBorder);
        break;
      case SamplerAddressMode::MirrorClampToEdge:
        sampler_create_info.setAddressModeW(
            vk::SamplerAddressMode::eMirrorClampToEdge);
        break;
      default:
        break;
    }

    switch (sampler_info.compare_op) {
      case SamplerCompareOp::Never:
        sampler_create_info.setCompareOp(vk::CompareOp::eNever);
        break;
      case SamplerCompareOp::Less:
        sampler_create_info.setCompareOp(vk::CompareOp::eLess);
        break;
      case SamplerCompareOp::Equal:
        sampler_create_info.setCompareOp(vk::CompareOp::eEqual);
        break;
      case SamplerCompareOp::LessOrEqual:
        sampler_create_info.setCompareOp(vk::CompareOp::eLessOrEqual);
        break;
      case SamplerCompareOp::Greater:
        sampler_create_info.setCompareOp(vk::CompareOp::eGreater);
        break;
      case SamplerCompareOp::NotEqual:
        sampler_create_info.setCompareOp(vk::CompareOp::eNotEqual);
        break;
      case SamplerCompareOp::GreaterOrEqual:
        sampler_create_info.setCompareOp(vk::CompareOp::eGreaterOrEqual);
        break;
      case SamplerCompareOp::Always:
        sampler_create_info.setCompareOp(vk::CompareOp::eAlways);
        break;
      default:
        break;
    }

    switch (sampler_info.border_color) {
      case SamplerBorderColor::FloatTransparentBlack:
        sampler_create_info.setBorderColor(
            vk::BorderColor::eFloatTransparentBlack);
        break;
      case SamplerBorderColor::IntTransparentBlack:
        sampler_create_info.setBorderColor(
            vk::BorderColor::eIntTransparentBlack);
        break;
      case SamplerBorderColor::FloatOpaqueBlack:
        sampler_create_info.setBorderColor(vk::BorderColor::eFloatOpaqueBlack);
        break;
      case SamplerBorderColor::IntOpaqueBlack:
        sampler_create_info.setBorderColor(vk::BorderColor::eIntOpaqueBlack);
        break;
      case SamplerBorderColor::FloatOpaqueWhite:
        sampler_create_info.setBorderColor(vk::BorderColor::eFloatOpaqueWhite);
        break;
      case SamplerBorderColor::IntOpaqueWhite:
        sampler_create_info.setBorderColor(vk::BorderColor::eIntOpaqueWhite);
        break;
      default:
        break;
    }
  }

  sampler_create_info.setAnisotropyEnable(sampler_info.anisotropy_enable);
  sampler_create_info.setMaxAnisotropy(sampler_info.max_anisotropy);
  sampler_create_info.setCompareEnable(sampler_info.compare_enable);
  sampler_create_info.setMipLodBias(sampler_info.mip_lod_bias);
  sampler_create_info.setMinLod(sampler_info.min_lod);
  sampler_create_info.setMaxLod(sampler_info.max_lod);
  sampler_create_info.setUnnormalizedCoordinates(
      sampler_info.unnormalized_coordinates);

  m_ptrSampler =
      ptr_context->getDevice()->getLogicalDevice()->createSamplerUnique(
          sampler_create_info);
}

hpxc::gpu::Sampler::~Sampler() {}
