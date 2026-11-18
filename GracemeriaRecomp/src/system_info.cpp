// System information at the top of the log (CPU, GPU, OS, RAM and graphics driver). Read once at startup,
// a few milliseconds: nothing here touches Vulkan (the renderer logs its own device with its driver further down).
#include "system_info.h"

#include "log.h"

#include <cpuid.h>
#include <cstring>
#include <format>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <dxgi.h>
#else
#include <fstream>
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#endif

namespace ac6::system_info {
namespace {

std::string Trim(std::string s)
{
    const size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

std::string Gigabytes(uint64_t bytes)
{
    return std::format("{:.1f} GB", double(bytes) / (1024.0 * 1024.0 * 1024.0));
}

std::string CpuBrand()
{
    unsigned regs[12] = {};
    unsigned a, b, c, d;
    if (!__get_cpuid(0x80000000, &a, &b, &c, &d) || a < 0x80000004)
        return "unknown CPU";
    for (unsigned i = 0; i < 3; i++)
        __get_cpuid(0x80000002 + i, &regs[i * 4], &regs[i * 4 + 1], &regs[i * 4 + 2], &regs[i * 4 + 3]);
    char brand[49] = {};
    std::memcpy(brand, regs, 48);
    // Some brands pad with runs of spaces in the middle: one space each.
    std::string out;
    for (const char* p = brand; *p; p++)
        if (*p != ' ' || (!out.empty() && out.back() != ' '))
            out += *p;
    return Trim(out);
}

bool HasAvx2()
{
    unsigned a, b, c, d;
    return __get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & (1u << 5));
}

#if defined(_WIN32)

std::string Narrow(const wchar_t* w)
{
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? size_t(n - 1) : 0, '\0');
    if (n > 1)
        WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

std::string RegString(const wchar_t* name)
{
    wchar_t buf[256] = {};
    DWORD size = sizeof(buf);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", name, RRF_RT_REG_SZ, nullptr,
            buf, &size) != ERROR_SUCCESS)
        return {};
    return Narrow(buf);
}

std::string OsName()
{
    // RtlGetVersion: the real version (GetVersionEx lies to programs without a manifest).
    struct OsVersion { ULONG size, major, minor, build, platform; WCHAR csd[128]; } v = { sizeof(OsVersion) };
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (auto rtl = ntdll ? reinterpret_cast<LONG(WINAPI*)(OsVersion*)>(GetProcAddress(ntdll, "RtlGetVersion")) : nullptr)
        rtl(&v);
    DWORD ubr = 0, size = sizeof(ubr);
    RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", L"UBR", RRF_RT_REG_DWORD, nullptr,
        &ubr, &size);
    // ProductName still says "Windows 10" on Windows 11: build 22000 and later is 11.
    std::string product = RegString(L"ProductName");
    if (product.empty())
        product = "Windows";
    if (v.build >= 22000)
        if (const size_t at = product.find("Windows 10"); at != std::string::npos)
            product.replace(at, 10, "Windows 11");
    const std::string release = RegString(L"DisplayVersion");
    std::string os = std::format("{}{} ({}.{}.{}.{})", product, release.empty() ? "" : " " + release, v.major, v.minor,
        v.build, ubr);
    // Wine / Proton: its version and the host system.
    if (auto wine = ntdll ? reinterpret_cast<const char*(CDECL*)()>(GetProcAddress(ntdll, "wine_get_version")) : nullptr)
    {
        os += std::format(", under Wine {}", wine());
        if (auto host = reinterpret_cast<void(CDECL*)(const char**, const char**)>(GetProcAddress(ntdll, "wine_get_host_version")))
        {
            const char *sys = nullptr, *rel = nullptr;
            host(&sys, &rel);
            if (sys)
                os += std::format(" on {} {}", sys, rel ? rel : "");
        }
    }
    return os;
}

std::pair<unsigned, unsigned> CoresThreads()
{
    unsigned cores = 0;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<uint8_t> buf(len);
    if (len && GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()), &len))
        for (DWORD at = 0; at < len; at += reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + at)->Size)
            cores++;
    return { cores, unsigned(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS)) };
}

void LogMemory()
{
    MEMORYSTATUSEX m = { sizeof(m) };
    if (GlobalMemoryStatusEx(&m))
        log::Info("System: RAM {} ({} free)", Gigabytes(m.ullTotalPhys), Gigabytes(m.ullAvailPhys));
}

// The GPUs as Windows lists them, with the installed driver's version: NVIDIA's own number from the last digits of
// the Windows one (32.0.15.6636 is 566.36), the Windows version for the others (AMD's Adrenalin number is not in it).
void LogGpus()
{
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) || !factory)
    {
        log::Info("System: GPU not listed (DXGI unavailable)");
        return;
    }
    // Each GPU once: Windows can list one more than once (a VM's partitioned GPU, one entry per output).
    std::set<std::string> logged;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++)
    {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
        {
            std::string driver = "unknown";
            LARGE_INTEGER umd{};
            if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd)))
            {
                const unsigned a = HIWORD(umd.HighPart), b = LOWORD(umd.HighPart), c = HIWORD(umd.LowPart),
                    d = LOWORD(umd.LowPart);
                driver = std::format("{}.{}.{}.{}", a, b, c, d);
                if (desc.VendorId == 0x10DE)
                    driver = std::format("{}.{:02} ({})", (c % 10) * 100 + d / 100, d % 100, driver);
            }
            std::string line = std::format("System: GPU {}, {} video memory, driver {}", Narrow(desc.Description),
                Gigabytes(desc.DedicatedVideoMemory), driver);
            if (logged.insert(line).second)
                log::Info("{}", line);
        }
        adapter->Release();
    }
    factory->Release();
}

#else

std::string OsName()
{
    std::string name;
    std::ifstream f("/etc/os-release");
    for (std::string line; std::getline(f, line);)
        if (line.rfind("PRETTY_NAME=", 0) == 0)
        {
            name = line.substr(12);
            if (name.size() >= 2 && name.front() == '"' && name.back() == '"')
                name = name.substr(1, name.size() - 2);
        }
    utsname u{};
    uname(&u);
    return std::format("{} ({} {})", name.empty() ? "Linux" : name, u.sysname, u.release);
}

std::pair<unsigned, unsigned> CoresThreads()
{
    std::set<std::pair<std::string, std::string>> cores;
    std::string physical;
    std::ifstream f("/proc/cpuinfo");
    for (std::string line; std::getline(f, line);)
    {
        const size_t colon = line.find(':');
        if (colon == std::string::npos)
            continue;
        const std::string key = Trim(line.substr(0, colon)), value = Trim(line.substr(colon + 1));
        if (key == "physical id")
            physical = value;
        else if (key == "core id")
            cores.insert({ physical, value });
    }
    return { unsigned(cores.size()), std::thread::hardware_concurrency() };
}

void LogMemory()
{
    uint64_t total = 0, available = 0;
    std::ifstream f("/proc/meminfo");
    for (std::string key; f >> key;)
    {
        uint64_t kb = 0;
        f >> kb;
        if (key == "MemTotal:")
            total = kb * 1024;
        else if (key == "MemAvailable:")
            available = kb * 1024;
        f.ignore(64, '\n');
    }
    if (total)
        log::Info("System: RAM {} ({} free)", Gigabytes(total), Gigabytes(available));
}

void LogGpus()
{
    // No driver query outside Vulkan here: the Vulkan device line below names the GPU and its driver.
}

#endif

} // namespace

void Log()
{
    log::Info("System: {}", OsName());
    const auto [cores, threads] = CoresThreads();
    log::Info("System: CPU {}, {} cores / {} threads{}", CpuBrand(), cores, threads, HasAvx2() ? ", AVX2" : ", no AVX2");
    LogMemory();
    LogGpus();
}

std::string DriverVersion(uint32_t vendorId, uint64_t v)
{
    const uint32_t x = uint32_t(v);
    if (vendorId == 0x10DE)   // NVIDIA: 10.8.8.6 bits
        return std::format("{}.{:02}", (x >> 22) & 0x3FF, (x >> 14) & 0xFF);
#if defined(_WIN32)
    if (vendorId == 0x8086)   // Intel's Windows driver: 18.14 bits
        return std::format("{}.{}", x >> 14, x & 0x3FFF);
#endif
    return std::format("{}.{}.{}", x >> 22, (x >> 12) & 0x3FF, x & 0xFFF);
}

} // namespace ac6::system_info
