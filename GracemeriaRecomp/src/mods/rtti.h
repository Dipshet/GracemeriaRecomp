#pragma once
// The game's C++ classes from its MSVC RTTI (in the image, big-endian): every polymorphic class's name and vtables, so
// mods can tell an object's class from its vtable and name virtual functions (Class::vfN).
//
// The image is scanned once, on first use: type descriptors (".?AV<name>@@" / ".?AU", the vtable pointer 8 bytes before),
// complete object locators (signature 0, the type descriptor at +12) and vtables (the locator in the word before
// slot 0). A class has one vtable per polymorphic base it's laid out with (`offset`: that base's place in the object).
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ac6::mods::rtti {

struct Vtable
{
    uint32_t address = 0;   // slot 0
    uint32_t offset = 0;    // the subobject's offset in the complete object (0: the primary vtable)
    uint32_t slots = 0;     // entries pointing at the game's functions
};

struct Class
{
    std::string name;       // ACE6::CAce6CameraManager (templates keep MSVC's spelling)
    std::string mangled;    // .?AVCAce6CameraManager@ACE6@@
    uint32_t typeDescriptor = 0;
    std::vector<Vtable> vtables;   // the primary (offset 0) first
};

const std::vector<Class>& Classes();
// By name (either spelling). Null: none.
const Class* Find(std::string_view name);
// The class of the object at `object` (its first word a vtable), or null. `offset`: which subobject's vtable it was.
const Class* Of(uint32_t object, uint32_t* offset = nullptr);
// MSVC's ".?AVName@Space@@" as "Space::Name".
std::string Demangle(std::string_view mangled);

} // namespace ac6::mods::rtti
