#include "../gpu.hpp"

hpxc::gpu::Context::Context(
    std::shared_ptr<gpu_ui_connection::WindowSurface> ptr_window_surface) {
  // Initialize Vulkan.hpp
  {
    static vk::DynamicLoader dl;
    auto vk_get_instance_proc_addr =
        dl.getProcAddress<::PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_get_instance_proc_addr);
  }

  // Create Vulkan instance
  {
    vk::ApplicationInfo app_info("hephics", HEPHICS_VK_VERSION, "hephics",
                                 HEPHICS_VK_VERSION, HEPHICS_VK_VERSION);
    uint32_t extension_count = 0U;
    const auto glfw_extensions =
        glfwGetRequiredInstanceExtensions(&extension_count);
    std::vector<const char*> extensions = {glfw_extensions,
                                           glfw_extensions + extension_count};
#ifndef HEPHICS_DEBUG
    vk::InstanceCreateInfo create_info({}, &app_info, {}, extensions);
    m_ptrInstance = vk::createInstanceUnique(create_info, nullptr);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*m_ptrInstance);
#else
    extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    extensions.emplace_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);

    m_ptrMessenger = std::make_unique<debug::Messenger>();
    m_ptrInstance = m_ptrMessenger->createDebugInstance(app_info, extensions);
    if (!m_ptrInstance) {
      return;
    }
#endif
  }

  if (ptr_window_surface) {
    // Create window surface
    m_ptrWindowSurface = ptr_window_surface;
    m_ptrWindowSurface->constructSurface(m_ptrInstance);

    // Create Vulkan device
    m_ptrDevice = std::make_unique<Device>(m_ptrInstance,
                                           m_ptrWindowSurface->getSurface());
  } else {
    // Create Vulkan device
    m_ptrDevice =
        std::make_unique<Device>(m_ptrInstance, vk::UniqueSurfaceKHR(nullptr));
  }

#ifdef HEPHICS_DEBUG
  m_ptrDevice->constructLogicalDevice(m_ptrMessenger);
#else
  m_ptrDevice->constructLogicalDevice();
#endif

  m_isInitialized = true;
}

hpxc::gpu::Context::~Context() {
  m_ptrDevice.release();
  m_ptrInstance.release();
#ifdef HEPHICS_DEBUG
  m_ptrMessenger.release();
#endif
}
