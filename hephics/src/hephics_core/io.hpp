/*
This header file is part of the Hephics project.
This header file is used to manage the asset file input/output.
  shaders, models, etc...
*/

#pragma once

namespace hpxc {

namespace io {

/// <summary>
/// This namespace is used to read and write shader code.
/// </summary>
namespace shader {

/// <summary>
/// Read shader code from glsl file (ex> .vert, .frag, etc...)
/// </summary>
/// <param name="file_path"></param>
/// <returns>spirv-binary</returns>
std::vector<uint32_t> readText(const std::string& file_path);

/// <summary>
/// Read spirv-binary from shader binary file (.spv).
/// </summary>
/// <param name="file_path"></param>
/// <returns>spirv-binary</returns>
std::vector<uint32_t> readBinary(const std::string& file_path);

/// <summary>
/// Read shader.
/// </summary>
/// <param name="file_path">path's extension
///   .spv, .vert, .frag, etc...
/// </param>
/// <returns>spirv-binary</returns>
std::vector<uint32_t> read(const std::string& file_path);

/// <summary>
/// Write shader binary.
/// </summary>
/// <param name="file_path">output file path</param>
/// <param name="shader_binary">spirv-binary written</param>
void write(const std::string& file_path,
           const std::vector<uint32_t>& shader_binary);

}  // namespace shader

}  // namespace io
}  // namespace hpxc
