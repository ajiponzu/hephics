/*
This header file is part of the Hephics project.
This file contains the functions for convert Hephics enum to Vulkan API's enum.
*/

#pragma once

#include "../gpu.hpp"

namespace vk_helper {

vk::MemoryPropertyFlags getMemoryPropertyFlags(
    const hpxc::MemoryUsage memory_usage);

vk::AccessFlagBits getAccessFlagBits(const hpxc::AccessFlag access_flag);

vk::PipelineStageFlagBits getPipelineStageFlagBits(
    const hpxc::PipelineStage stage);

vk::ImageLayout getImageLayout(const hpxc::ImageLayout image_layout);

vk::Format getImageFormat(const hpxc::ImageFormat image_format);

vk::ImageAspectFlags getImageAspectFlags(const hpxc::ImageAspect image_aspect);

vk::ShaderStageFlags getShaderStageFlagBits(
    const hpxc::ShaderStage shader_stage);
}  // namespace vk_helper
