#include "../gpu_ui.hpp"

hpxc::gpu_ui_connection::WindowSurface::WindowSurface(
    GLFWwindow* const ptr_window) {
  m_ptrWindow = ptr_window;
}

hpxc::gpu_ui_connection::WindowSurface::~WindowSurface() {
  m_ptrSurface.release();
  glfwDestroyWindow(m_ptrWindow);
}

void hpxc::gpu_ui_connection::WindowSurface::constructSurface(
    const vk::UniqueInstance& ptr_instance) {
  VkSurfaceKHR surface;
  if (glfwCreateWindowSurface(VkInstance(ptr_instance.get()), m_ptrWindow,
                              nullptr, &surface) != ::VkResult::VK_SUCCESS) {
    m_ptrSurface = vk::UniqueSurfaceKHR(nullptr);
    return;
  }

  m_ptrSurface = vk::UniqueSurfaceKHR(surface, {ptr_instance.get()});
}
