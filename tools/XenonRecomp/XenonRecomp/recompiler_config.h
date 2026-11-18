#pragma once

#include <unordered_set>

struct RecompilerSwitchTable
{
    uint32_t r;
    std::vector<uint32_t> labels;
};

struct RecompilerMidAsmHook
{
    std::string name;
    std::vector<std::string> registers;

    bool ret = false;
    bool returnOnTrue = false;
    bool returnOnFalse = false;

    uint32_t jumpAddress = 0;
    uint32_t jumpAddressOnTrue = 0;
    uint32_t jumpAddressOnFalse = 0;

    bool afterInstruction = false;
};

struct RecompilerConfig
{
    std::string directoryPath;
    std::string filePath;
    std::string patchFilePath;
    std::string patchedFilePath;
    std::string outDirectoryPath;
    std::string switchTableFilePath;
    std::unordered_map<uint32_t, RecompilerSwitchTable> switchTables;
    bool skipLr = false;
    bool ctrAsLocalVariable = false;
    bool xerAsLocalVariable = false;
    bool reservedRegisterAsLocalVariable = false;
    bool skipMsr = false;
    bool crRegistersAsLocalVariables = false;
    bool nonArgumentRegistersAsLocalVariables = false;
    bool nonVolatileRegistersAsLocalVariables = false;
    uint32_t restGpr14Address = 0;
    uint32_t saveGpr14Address = 0;
    uint32_t restFpr14Address = 0;
    uint32_t saveFpr14Address = 0;
    uint32_t restVmx14Address = 0;
    uint32_t saveVmx14Address = 0;
    uint32_t restVmx64Address = 0;
    uint32_t saveVmx64Address = 0;
    uint32_t longJmpAddress = 0;
    uint32_t setJmpAddress = 0;
    std::unordered_map<uint32_t, uint32_t> functions;
    std::unordered_map<uint32_t, uint32_t> invalidInstructions;
    // AC6: functions that keep the non-volatile registers in the context despite non_volatile_as_local (they
    // share registers outside the ABI: SEH funclets use their parent's), together with the save/restore helpers.
    std::unordered_set<uint32_t> contextRegisterFunctions;
    std::unordered_map<uint32_t, RecompilerMidAsmHook> midAsmHooks;

    void Load(const std::string_view& configFilePath);
};
