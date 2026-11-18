// Updates from GitHub releases (update/updater.h).
#include "update/updater.h"

#include "log.h"
#include "platform/platform.h"
#include "settings.h"

#include <algorithm>
#include <cerrno>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <winhttp.h>
#ifndef WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY
#define WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY 4
#endif
#else
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

#ifndef AC6_VERSION
#define AC6_VERSION "0.0.0"
#endif

namespace ac6::update {
namespace {

constexpr const char* kRepository = "Dipshet/GracemeriaRecomp";
constexpr const char* kReplaced = "update_replaced.txt";   // the files the last update renamed to *.old, one a line
// While files are being swapped: each move before it is made ("R <target>\t<old>" for a file renamed aside, "N <target>"
// for a new one). Left behind by a swap that was cut off (the game closed or crashed mid-install), it tells the next start
// to finish the swap (FinishInterruptedSwap).
constexpr const char* kSwapping = "update_swapping.txt";
#if defined(_WIN32) && defined(AC6_X86_V2)
constexpr const char* kAssetSuffix = "-windows-x64-compat.zip";   // (the compatibility version updates from its own)
#elif defined(_WIN32)
constexpr const char* kAssetSuffix = "-windows-x64.zip";
#else
constexpr const char* kAssetSuffix = "-linux-x64.tar.gz";
#endif

std::mutex s_mutex;
Status s_status;
std::string s_assetUrl, s_assetName, s_assetSha256;
uint64_t s_assetSize = 0;
std::atomic<bool> s_busy{ false };

void Set(State state, std::string error = {})
{
    std::lock_guard lock(s_mutex);
    s_status.state = state;
    s_status.error = std::move(error);
    if (state != State::Downloading)
        s_status.progress = -1.0f;
}

// --- Versions -------------------------------------------------------------------------------------------------------

// "v1.2.3", "1.2" -> {1, 2, 3} / {1, 2, 0}. Anything after the numbers (a pre-release suffix) is ignored.
std::array<int, 3> ParseVersion(std::string_view v)
{
    std::array<int, 3> out{};
    size_t at = 0;
    while (at < v.size() && !(v[at] >= '0' && v[at] <= '9'))
        at++;
    for (int part = 0; part < 3 && at < v.size(); part++)
    {
        int n = 0;
        bool digits = false;
        while (at < v.size() && v[at] >= '0' && v[at] <= '9')
        {
            n = n * 10 + (v[at++] - '0');
            digits = true;
        }
        if (!digits)
            break;
        out[size_t(part)] = n;
        if (at < v.size() && v[at] == '.')
            at++;
        else
            break;
    }
    return out;
}

bool Newer(std::string_view candidate, std::string_view current) { return ParseVersion(candidate) > ParseVersion(current); }

// --- JSON (only what the releases API's answers need) -----------------------------------------------------------------

struct JsonMember;

struct Json
{
    enum class Type { Null, Bool, Number, String, Array, Object } type = Type::Null;
    std::string text;   // a string's value, a number's digits
    bool boolean = false;
    std::vector<Json> items;
    // (A struct of its own, not pair<string, Json>: libstdc++ 15 can't instantiate that pair inside Json.)
    std::vector<JsonMember> members;

    const Json* Get(std::string_view key) const;
    std::string String(std::string_view key) const;
};

struct JsonMember
{
    std::string key;
    Json value;
};

const Json* Json::Get(std::string_view key) const
{
    for (const JsonMember& m : members)
        if (m.key == key)
            return &m.value;
    return nullptr;
}

std::string Json::String(std::string_view key) const
{
    const Json* v = Get(key);
    return v && v->type == Type::String ? v->text : std::string();
}

class JsonParser
{
public:
    explicit JsonParser(std::string_view text) : t(text) {}
    bool Parse(Json& out)
    {
        if (!Value(out, 0))
            return false;
        Space();
        return at == t.size();
    }

private:
    std::string_view t;
    size_t at = 0;

    void Space()
    {
        while (at < t.size() && (t[at] == ' ' || t[at] == '\n' || t[at] == '\r' || t[at] == '\t'))
            at++;
    }
    bool Literal(std::string_view word)
    {
        if (t.substr(at, word.size()) != word)
            return false;
        at += word.size();
        return true;
    }
    static void Utf8(std::string& out, uint32_t c)
    {
        if (c < 0x80)
            out += char(c);
        else if (c < 0x800)
        {
            out += char(0xC0 | (c >> 6));
            out += char(0x80 | (c & 0x3F));
        }
        else if (c < 0x10000)
        {
            out += char(0xE0 | (c >> 12));
            out += char(0x80 | ((c >> 6) & 0x3F));
            out += char(0x80 | (c & 0x3F));
        }
        else
        {
            out += char(0xF0 | (c >> 18));
            out += char(0x80 | ((c >> 12) & 0x3F));
            out += char(0x80 | ((c >> 6) & 0x3F));
            out += char(0x80 | (c & 0x3F));
        }
    }
    bool Hex4(uint32_t& c)
    {
        if (at + 4 > t.size())
            return false;
        c = 0;
        for (int i = 0; i < 4; i++)
        {
            const char h = t[at++];
            c <<= 4;
            if (h >= '0' && h <= '9')
                c |= uint32_t(h - '0');
            else if (h >= 'a' && h <= 'f')
                c |= uint32_t(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F')
                c |= uint32_t(h - 'A' + 10);
            else
                return false;
        }
        return true;
    }
    bool String(std::string& out)
    {
        if (at >= t.size() || t[at] != '"')
            return false;
        at++;
        while (at < t.size() && t[at] != '"')
        {
            const char c = t[at++];
            if (c != '\\')
            {
                out += c;
                continue;
            }
            if (at >= t.size())
                return false;
            const char e = t[at++];
            switch (e)
            {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u':
            {
                uint32_t c1 = 0;
                if (!Hex4(c1))
                    return false;
                if (c1 >= 0xD800 && c1 < 0xDC00 && Literal("\\u"))
                {
                    uint32_t c2 = 0;
                    if (!Hex4(c2))
                        return false;
                    c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
                }
                Utf8(out, c1);
                break;
            }
            default: return false;
            }
        }
        if (at >= t.size())
            return false;
        at++;
        return true;
    }
    bool Value(Json& out, int depth)
    {
        if (depth > 64)
            return false;
        Space();
        if (at >= t.size())
            return false;
        const char c = t[at];
        if (c == '{')
        {
            out.type = Json::Type::Object;
            at++;
            Space();
            if (at < t.size() && t[at] == '}')
            {
                at++;
                return true;
            }
            for (;;)
            {
                Space();
                std::string key;
                if (!String(key))
                    return false;
                Space();
                if (at >= t.size() || t[at] != ':')
                    return false;
                at++;
                Json v;
                if (!Value(v, depth + 1))
                    return false;
                out.members.push_back({ std::move(key), std::move(v) });
                Space();
                if (at < t.size() && t[at] == ',')
                {
                    at++;
                    continue;
                }
                if (at < t.size() && t[at] == '}')
                {
                    at++;
                    return true;
                }
                return false;
            }
        }
        if (c == '[')
        {
            out.type = Json::Type::Array;
            at++;
            Space();
            if (at < t.size() && t[at] == ']')
            {
                at++;
                return true;
            }
            for (;;)
            {
                Json v;
                if (!Value(v, depth + 1))
                    return false;
                out.items.push_back(std::move(v));
                Space();
                if (at < t.size() && t[at] == ',')
                {
                    at++;
                    continue;
                }
                if (at < t.size() && t[at] == ']')
                {
                    at++;
                    return true;
                }
                return false;
            }
        }
        if (c == '"')
        {
            out.type = Json::Type::String;
            return String(out.text);
        }
        if (Literal("true"))
        {
            out.type = Json::Type::Bool;
            out.boolean = true;
            return true;
        }
        if (Literal("false"))
        {
            out.type = Json::Type::Bool;
            return true;
        }
        if (Literal("null"))
            return true;
        const size_t start = at;
        while (at < t.size() && (std::strchr("+-.eE", t[at]) || (t[at] >= '0' && t[at] <= '9')))
            at++;
        if (at == start)
            return false;
        out.type = Json::Type::Number;
        out.text = std::string(t.substr(start, at - start));
        return true;
    }
};

// --- SHA-256 (FIPS 180-4) ------------------------------------------------------------------------------------------------

class Sha256
{
public:
    void Update(const uint8_t* data, size_t size)
    {
        length += size;
        while (size)
        {
            const size_t n = std::min(size, size_t(64) - fill);
            std::memcpy(block + fill, data, n);
            fill += n;
            data += n;
            size -= n;
            if (fill == 64)
            {
                Compress();
                fill = 0;
            }
        }
    }
    std::string HexDigest()
    {
        const uint64_t bits = length * 8;
        const uint8_t pad = 0x80, zero = 0;
        Update(&pad, 1);
        while (fill != 56)
            Update(&zero, 1);
        uint8_t big[8];
        for (int i = 0; i < 8; i++)
            big[i] = uint8_t(bits >> (56 - 8 * i));
        Update(big, 8);
        static const char* hex = "0123456789abcdef";
        std::string out;
        for (uint32_t v : h)
            for (int i = 28; i >= 0; i -= 4)
                out += hex[(v >> i) & 0xF];
        return out;
    }

private:
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    uint8_t block[64] = {};
    size_t fill = 0;
    uint64_t length = 0;

    static uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void Compress()
    {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
            0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
            0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
            0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2 };
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = uint32_t(block[4 * i]) << 24 | uint32_t(block[4 * i + 1]) << 16 | uint32_t(block[4 * i + 2]) << 8 | block[4 * i + 3];
        for (int i = 16; i < 64; i++)
        {
            const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++)
        {
            const uint32_t t1 = hh + (Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }
};

bool FileSha256(const std::filesystem::path& path, std::string& hex, uint64_t& size)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    Sha256 sha;
    std::vector<char> buffer(1 << 20);
    size = 0;
    while (in)
    {
        in.read(buffer.data(), std::streamsize(buffer.size()));
        const auto n = in.gcount();
        if (n <= 0)
            break;
        sha.Update(reinterpret_cast<const uint8_t*>(buffer.data()), size_t(n));
        size += uint64_t(n);
    }
    hex = sha.HexDigest();
    return true;
}

// --- HTTPS and processes ------------------------------------------------------------------------------------------------

std::string UserAgent() { return std::string("GracemeriaRecomp/") + AC6_VERSION; }

using Progress = std::function<void(uint64_t done, uint64_t total)>;

#if defined(_WIN32)

std::wstring Wide(const std::string& s)
{
    if (s.empty())
        return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring out(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), out.data(), n);
    return out;
}

// GET `url` (redirects followed) into `file`, or into `body` when `file` is empty.
bool Get(const std::string& url, const std::filesystem::path& file, std::string* body, const char* accept, const Progress& progress,
    std::string& error)
{
    const std::wstring wurl = Wide(url);
    URL_COMPONENTS parts{};
    parts.dwStructSize = sizeof(parts);
    parts.dwHostNameLength = DWORD(-1);
    parts.dwUrlPathLength = DWORD(-1);
    parts.dwExtraInfoLength = DWORD(-1);
    if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &parts))
    {
        error = "bad address";
        return false;
    }
    const std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
    std::wstring path(parts.lpszUrlPath, parts.dwUrlPathLength);
    if (parts.lpszExtraInfo)
        path.append(parts.lpszExtraInfo, parts.dwExtraInfoLength);

    const std::wstring agent = Wide(UserAgent());
    HINTERNET session = WinHttpOpen(agent.c_str(), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session)   // (before Windows 8.1)
        session = WinHttpOpen(agent.c_str(), WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session)
    {
        error = "no network access";
        return false;
    }
    WinHttpSetTimeouts(session, 15000, 15000, 30000, 60000);
    HINTERNET connection = WinHttpConnect(session, host.c_str(), parts.nPort, 0);
    HINTERNET request = connection ? WinHttpOpenRequest(connection, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER,
                                         WINHTTP_DEFAULT_ACCEPT_TYPES, parts.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0)
                                   : nullptr;
    auto close = [&] {
        if (request)
            WinHttpCloseHandle(request);
        if (connection)
            WinHttpCloseHandle(connection);
        WinHttpCloseHandle(session);
    };
    const std::wstring headers = Wide(std::string("Accept: ") + accept + "\r\n");
    if (!request || !WinHttpSendRequest(request, headers.c_str(), DWORD(-1), WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request, nullptr))
    {
        error = "couldn't reach " + std::string(host.begin(), host.end());
        close();
        return false;
    }
    DWORD status = 0, size = sizeof(status);
    WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &size,
        WINHTTP_NO_HEADER_INDEX);
    if (status != 200)
    {
        error = "HTTP " + std::to_string(status);
        close();
        return false;
    }
    DWORD length = 0;
    size = sizeof(length);
    const uint64_t total = WinHttpQueryHeaders(request, WINHTTP_QUERY_CONTENT_LENGTH | WINHTTP_QUERY_FLAG_NUMBER,
                               WINHTTP_HEADER_NAME_BY_INDEX, &length, &size, WINHTTP_NO_HEADER_INDEX)
        ? length
        : 0;
    std::ofstream out;
    if (!file.empty())
    {
        out.open(file, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            error = "can't write " + platform::PathUtf8(file);
            close();
            return false;
        }
    }
    std::vector<char> buffer(1 << 16);
    uint64_t done = 0;
    for (;;)
    {
        DWORD read = 0;
        if (!WinHttpReadData(request, buffer.data(), DWORD(buffer.size()), &read))
        {
            error = "the download stopped";
            close();
            return false;
        }
        if (read == 0)
            break;
        if (body)
            body->append(buffer.data(), read);
        else
            out.write(buffer.data(), read);
        done += read;
        if (progress)
            progress(done, total);
    }
    close();
    if (out.is_open())
    {
        out.close();
        if (!out)
        {
            error = "can't write " + platform::PathUtf8(file);
            return false;
        }
    }
    return true;
}

bool Run(const std::vector<std::string>& args, std::string& error)
{
    std::wstring line;
    for (const std::string& a : args)
    {
        if (!line.empty())
            line += L' ';
        line += L'"' + Wide(a) + L'"';
    }
    STARTUPINFOW startup{ sizeof(startup) };
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(nullptr, line.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process))
    {
        error = "couldn't run " + args[0];
        return false;
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    if (code != 0)
        error = args[0] + " failed (" + std::to_string(code) + ")";
    return code == 0;
}

std::string TarPath()
{
    wchar_t system[MAX_PATH];
    const UINT n = GetSystemDirectoryW(system, MAX_PATH);
    return platform::PathUtf8(std::filesystem::path(std::wstring(system, n)) / L"tar.exe");   // (Windows 10 1803 and later)
}

#else

bool Run(const std::vector<std::string>& args, std::string& error)
{
    std::vector<char*> argv;
    std::vector<std::string> copy = args;
    for (std::string& a : copy)
        argv.push_back(a.data());
    argv.push_back(nullptr);
    pid_t pid = 0;
    if (posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), environ) != 0)
    {
        error = "couldn't run " + args[0];
        return false;
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0)
        if (errno != EINTR)
        {
            error = args[0] + " couldn't be waited for";
            return false;
        }
    const bool ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (!ok)
        error = args[0] + " failed";
    return ok;
}

// curl does the HTTPS (it is on every desktop distribution and SteamOS).
bool Get(const std::string& url, const std::filesystem::path& file, std::string* body, const char* accept, const Progress&,
    std::string& error)
{
    // The release JSON into a file only this process made (mkstemp: no name another user can claim first).
    std::filesystem::path target = file;
    if (file.empty())
    {
        std::error_code ec;
        std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
        if (ec)
            dir = "/tmp";
        std::string name = (dir / "ac6_update_XXXXXX").string();
        const int fd = mkstemp(name.data());
        if (fd < 0)
        {
            error = "no temporary file for the download";
            return false;
        }
        close(fd);
        target = name;
    }
    const bool ok = Run({ "curl", "-fsSL", "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "15", "--max-time", "1800",
                            "-A", UserAgent(), "-H",
                            std::string("Accept: ") + accept, "-o", target.string(), url },
        error);
    if (ok && body)
    {
        std::ifstream in(target, std::ios::binary);
        body->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    if (file.empty())
    {
        std::error_code ec;
        std::filesystem::remove(target, ec);
    }
    if (!ok)
        error = "the download failed (curl)";
    return ok;
}

std::string TarPath() { return "tar"; }

#endif

// --- The work -----------------------------------------------------------------------------------------------------------

std::filesystem::path Folder() { return platform::ExecutableDirectory(); }

// Tests: AC6_UPDATE_API=<url> asks another address for the release, such as a local server with a test release.
// When it is set, the check also runs in scripted runs. AC6_UPDATE_AUTOINSTALL=1 installs what it finds without the
// overlay. Both only from the environment: never from the settings file's [env], since a shared settings file must
// not be able to point updates at another server.
const char* TestSwitch(const char* name)
{
    const char* v = std::getenv(name);
    return v && *v && !settings::SetByFile(name) ? v : nullptr;
}

const char* TestApi() { return TestSwitch("AC6_UPDATE_API"); }

void InstallWork();

bool CheckSetting()
{
    const std::string v = settings::Get("AC6_CHECK_UPDATES");
    return v != "0" && v != "false";
}

void CheckWork()
{
    std::string body, error;
    const std::string url = TestApi() ? std::string(TestApi()) : std::string("https://api.github.com/repos/") + kRepository + "/releases/latest";
    if (!Get(url, {}, &body, "application/vnd.github+json", {}, error))
    {
        log::Warn("Updates: checking {} failed: {}", url, error);
        Set(State::Failed, error == "HTTP 404" ? "no release published yet" : error);
        return;
    }
    Json release;
    if (!JsonParser(body).Parse(release) || release.type != Json::Type::Object)
    {
        log::Warn("Updates: GitHub's answer couldn't be read");
        Set(State::Failed, "GitHub's answer couldn't be read");
        return;
    }
    const std::string tag = release.String("tag_name");
    std::string latest = tag;
    if (!latest.empty() && (latest[0] == 'v' || latest[0] == 'V'))
        latest.erase(0, 1);
    {
        std::lock_guard lock(s_mutex);
        s_status.latest = latest;
    }
    if (!Newer(latest, AC6_VERSION))
    {
        log::Info("Updates: up to date ({}; the newest release is {})", AC6_VERSION, tag);
        Set(State::UpToDate);
        return;
    }
    const Json* assets = release.Get("assets");
    const Json* asset = nullptr;
    if (assets && assets->type == Json::Type::Array)
        for (const Json& a : assets->items)
        {
            const std::string name = a.String("name");
            if (name.size() >= std::strlen(kAssetSuffix) && name.ends_with(kAssetSuffix))
                asset = &a;
        }
    if (!asset)
    {
        log::Warn("Updates: release {} has no download for this system (*{})", tag, kAssetSuffix);
        Set(State::Failed, std::string("release ") + tag + " has no download for this system");
        return;
    }
    std::string digest = asset->String("digest");
    if (digest.rfind("sha256:", 0) != 0 || digest.size() != 7 + 64)
    {
        log::Warn("Updates: release {}'s download has no SHA-256 from GitHub: not installed", tag);
        Set(State::Failed, "the download has no checksum");
        return;
    }
    {
        std::lock_guard lock(s_mutex);
        s_assetUrl = asset->String("browser_download_url");
        s_assetName = asset->String("name");
        s_assetSha256 = digest.substr(7);
        const Json* size = asset->Get("size");
        s_assetSize = size && size->type == Json::Type::Number ? std::strtoull(size->text.c_str(), nullptr, 10) : 0;
    }
    log::Info("Updates: version {} is available ({}; this is {})", latest, s_assetName, AC6_VERSION);
    Set(State::Available);
    if (const char* v = TestSwitch("AC6_UPDATE_AUTOINSTALL"); v && *v == '1')
        InstallWork();
}

// The release's files moved in next to the executable. The ones replaced kept as <name>.old (a running executable can be
// renamed but not overwritten on Windows). All or nothing: on a failure the moved files go back.
bool Swap(const std::filesystem::path& staged, std::string& error)
{
    namespace fs = std::filesystem;
    const fs::path folder = Folder();
    std::vector<std::pair<fs::path, fs::path>> kept;    // target, its .old
    std::vector<fs::path> placed;
    std::ofstream journal(folder / kSwapping, std::ios::binary | std::ios::trunc);
    if (!journal)
    {
        error = "can't write to the game's folder";
        return false;
    }
    auto note = [&](const std::string& line) {
        journal << line << '\n';
        journal.flush();
        return bool(journal);
    };
    auto undo = [&] {
        std::error_code ec;
        for (const fs::path& p : placed)
            fs::remove(p, ec);
        for (const auto& [target, old] : kept)
            fs::rename(old, target, ec);
        journal.close();
        fs::remove(folder / kSwapping, ec);
    };
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(staged, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec))
    {
        std::error_code fileError;
        if (!it->is_regular_file(fileError))
            continue;
        const fs::path rel = fs::relative(it->path(), staged, ec);
        const fs::path target = folder / rel;
        fs::create_directories(target.parent_path(), ec);
        if (fs::exists(target, fileError))
        {
            fs::path old = target;
            old += ".old";
            fs::remove(old, ec);
            for (int n = 2; fs::exists(old, fileError) && n < 10; n++)   // (an .old still running)
            {
                old = target;
                old += ".old" + std::to_string(n);
                fs::remove(old, ec);
            }
            if (!note("R " + platform::PathUtf8(rel) + "\t" + platform::PathUtf8(fs::relative(old, folder, fileError))))
            {
                error = "can't write to the game's folder";
                undo();
                return false;
            }
            fs::rename(target, old, ec);
            if (ec)
            {
                error = "can't replace " + platform::PathUtf8(rel) + " (" + ec.message() + ")";
                undo();
                return false;
            }
            kept.emplace_back(target, old);
        }
        else if (!note("N " + platform::PathUtf8(rel)))
        {
            error = "can't write to the game's folder";
            undo();
            return false;
        }
        fs::rename(it->path(), target, ec);
        if (ec)
        {
            error = "can't write " + platform::PathUtf8(rel) + " (" + ec.message() + ")";
            undo();
            return false;
        }
        placed.push_back(target);
    }
    if (ec)
    {
        error = "can't read the unpacked files";
        undo();
        return false;
    }
    // The replaced files, for their removal at the next start (and only theirs). Then the swap is complete.
    {
        std::ofstream list(folder / kReplaced, std::ios::binary | std::ios::trunc);
        for (const auto& [target, old] : kept)
            list << platform::PathUtf8(fs::relative(old, folder, ec)) << '\n';
    }
    journal.close();
    fs::remove(folder / kSwapping, ec);
    return true;
}

void InstallWork()
{
    namespace fs = std::filesystem;
    std::string url, name, sha;
    uint64_t expectedSize = 0;
    {
        std::lock_guard lock(s_mutex);
        url = s_assetUrl;
        name = s_assetName;
        sha = s_assetSha256;
        expectedSize = s_assetSize;
    }
    const fs::path work = Folder() / "update";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work / "files", ec);
    if (ec)
    {
        Set(State::Failed, "can't write to the game's folder");
        return;
    }
    const fs::path archive = work / name;
    Set(State::Downloading);
    log::Info("Updates: downloading {}", url);
    std::string error;
    const bool got = Get(url, archive, nullptr, "application/octet-stream",
        [&](uint64_t done, uint64_t total) {
            std::lock_guard lock(s_mutex);
            const uint64_t all = total ? total : expectedSize;
            s_status.progress = all ? float(double(done) / double(all)) : -1.0f;
        },
        error);
    if (!got)
    {
        log::Warn("Updates: the download failed: {}", error);
        fs::remove_all(work, ec);
        Set(State::Failed, error);
        return;
    }
    Set(State::Installing);
    std::string hex;
    uint64_t size = 0;
    if (!FileSha256(archive, hex, size) || hex != sha || (expectedSize && size != expectedSize))
    {
        log::Warn("Updates: {} doesn't match GitHub's SHA-256 ({} bytes, {}; expected {}): not installed", name, size, hex, sha);
        fs::remove_all(work, ec);
        Set(State::Failed, "the download was damaged (checksum)");
        return;
    }
    if (!Run({ TarPath(), "-xf", platform::PathUtf8(archive), "-C", platform::PathUtf8(work / "files") }, error))
    {
        log::Warn("Updates: unpacking {} failed: {}", name, error);
        fs::remove_all(work, ec);
        Set(State::Failed, "couldn't unpack the download");
        return;
    }
    // A release archive: this executable's name inside, at the top.
    if (!fs::exists(work / "files" / platform::ExecutablePath().filename()))
    {
        log::Warn("Updates: {} isn't a release archive of this game (no {} at its top)", name,
            platform::PathUtf8(platform::ExecutablePath().filename()));
        fs::remove_all(work, ec);
        Set(State::Failed, "the download isn't a release of this game");
        return;
    }
    if (!Swap(work / "files", error))
    {
        log::Warn("Updates: installing failed: {}", error);
        fs::remove_all(work, ec);
        Set(State::Failed, error);
        return;
    }
    fs::remove_all(work, ec);
    log::Info("Updates: {} installed: it runs from the next start", name);
    Set(State::Installed);
}

void RunInBackground(void (*work)())
{
    bool expected = false;
    if (!s_busy.compare_exchange_strong(expected, true))
        return;
    std::thread([work] {
        try
        {
            work();
        }
        catch (const std::exception& e)
        {
            log::Error("Updates: stopped by an error ({})", e.what());
            Set(State::Failed, e.what());
        }
        s_busy = false;
    }).detach();
}

// A swap cut off (the game closed or crashed mid-install: the journal still there): finished from the release files still
// waiting in update/files (checked against GitHub's SHA-256 before the swap began). Finishing, not undoing: the
// executable starting now may already be the new one, and a running executable can't be put back.
void FinishInterruptedSwap()
{
    namespace fs = std::filesystem;
    const fs::path folder = Folder(), journalPath = folder / kSwapping, staged = folder / "update" / "files";
    std::error_code ec;
    if (!fs::exists(journalPath, ec))
        return;
    // The files already renamed aside, for the removal list (Swap writes a new journal and list).
    std::vector<std::string> olds;
    {
        std::ifstream in(journalPath, std::ios::binary);
        for (std::string line; std::getline(in, line);)
        {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (line.size() > 2 && line[0] == 'R')
                if (const size_t tab = line.find('\t'); tab != std::string::npos)
                    olds.push_back(line.substr(tab + 1));
        }
    }
    std::string error;
    if (!fs::is_directory(staged, ec) || !Swap(staged, error))
    {
        log::Warn("Updates: the last install was cut off and can't be finished ({}): download it again",
            error.empty() ? "its files are gone" : error);
        fs::remove(journalPath, ec);
        return;
    }
    {
        std::ofstream list(folder / kReplaced, std::ios::binary | std::ios::app);
        for (const std::string& old : olds)
            list << old << '\n';
    }
    fs::remove_all(folder / "update", ec);
    log::Info("Updates: the last install was cut off; it is finished now and runs from the next start");
}

// The files the last update replaced (<name>.old, .old2...), now that nothing runs from them: those it listed, and
// only names ending in .old or .old<digits> inside the game's folder (anything else in the list is left alone).
void RemoveOldFiles()
{
    namespace fs = std::filesystem;
    const fs::path folder = Folder(), list = folder / kReplaced;
    std::ifstream in(list, std::ios::binary);
    if (!in)
        return;
    std::string line;
    while (std::getline(in, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        const fs::path rel = fs::path(std::u8string(reinterpret_cast<const char8_t*>(line.data()), line.size()));
        const std::string ext = rel.extension().string();
        if (line.empty() || rel.is_absolute() || line.find("..") != std::string::npos || ext.rfind(".old", 0) != 0 ||
            !std::all_of(ext.begin() + 4, ext.end(), [](char c) { return c >= '0' && c <= '9'; }))
            continue;
        std::error_code ec;
        if (fs::is_regular_file(folder / rel, ec))
            fs::remove(folder / rel, ec);
    }
    in.close();
    std::error_code ec;
    fs::remove(list, ec);
}

} // namespace

const char* Version() { return AC6_VERSION; }

void Start(bool automatic)
{
#if defined(AC6_RELEASE)
    constexpr bool release = true;
#else
    constexpr bool release = false;
#endif
    if (release)
    {
        RemoveOldFiles();          // the last update's .old files
        FinishInterruptedSwap();   // an install cut off: finished (its .old files listed for the next start)
    }
    {
        std::lock_guard lock(s_mutex);
        s_status = {};
        s_status.state = release ? State::Idle : State::Off;
    }
    if (!release)
    {
        log::Info("Updates: version {}, not a release build: no update checks", AC6_VERSION);
        return;
    }
    if (TestApi())
        automatic = true;
    if (!automatic || !CheckSetting())
    {
        log::Info("Updates: version {}; not checked at start ({})", AC6_VERSION, automatic ? "check_updates is off" : "a scripted run");
        return;
    }
    Check();
}

void Check()
{
    {
        std::lock_guard lock(s_mutex);
        if (s_status.state == State::Off || s_status.state == State::Installed)
            return;
    }
    if (s_busy)
        return;
    Set(State::Checking);
    RunInBackground(CheckWork);
}

void Install()
{
    {
        std::lock_guard lock(s_mutex);
        if (s_status.state != State::Available)
            return;
    }
    if (s_busy)
        return;
    Set(State::Downloading);
    RunInBackground(InstallWork);
}

Status Get()
{
    std::lock_guard lock(s_mutex);
    return s_status;
}

} // namespace ac6::update
