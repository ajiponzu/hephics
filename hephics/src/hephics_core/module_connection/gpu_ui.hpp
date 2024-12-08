/*
This header file contains a module connecting connect gpu.hpp and ui.hpp.
*/

#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <concepts>
#include <type_traits>

namespace hpxc {
namespace gpu_ui_connection {

template <typename T>
concept GraphicalSizeConcept =
    std::is_arithmetic_v<T> && !std::is_same<T, bool>::value;

template <GraphicalSizeConcept T>
struct GraphicalSize {
  T width;
  T height;
  T depth;
};

/// <summary>
/// This class is vulkan surface wrapper.
/// And, this class is used to connect vulkan and glfw.
/// </summary>
class WindowSurface {
 private:
  GLFWwindow* m_ptrWindow;
  vk::UniqueSurfaceKHR m_ptrSurface;

 public:
  WindowSurface(GLFWwindow* const ptr_window);
  ~WindowSurface();

  auto getWindow() const { return m_ptrWindow; };
  const auto& getSurface() const { return m_ptrSurface; };

  void constructSurface(const vk::UniqueInstance& ptr_instance);
};

}  // namespace gpu_ui_connection
}  // namespace hpxc
