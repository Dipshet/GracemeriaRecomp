# FSR 3.1's compute passes, compiled

AMD FidelityFX SDK 1.1.4's FSR 3.1 upscaler shaders (`sdk/src/backends/vk/shaders/fsr3upscaler`, MIT: see
`LICENSE.txt`), compiled to SPIR-V with every permutation, as the headers AMD's Vulkan backend includes. They are made
by the SDK's own shader compiler driver, FidelityFX_SC, which runs on Windows only
(`tools/cmake/compile_ffx_shaders.cmake`): Windows builds compile them again from the SDK; other builds use these.
SPIR-V is the same on every platform, so they are byte for byte what a Windows build makes.

The port's patch to the SDK does not touch the shaders. Should the SDK or its shaders change, copy a Windows build's
`build/generated/ffx/*.h` here: a Windows build warns when the two differ.
