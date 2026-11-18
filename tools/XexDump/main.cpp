// Dumps a loaded XEX image (decrypted/decompressed) plus section and import info.
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <string>
#include <file.h>
#include <image.h>
#include <disasm.h>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("Usage: XexDump [input XEX] [output prefix]\n");
        return 1;
    }

    const auto file = LoadFile(argv[1]);
    auto image = Image::ParseImage(file.data(), file.size());
    if (image.data == nullptr)
    {
        printf("Failed to load image\n");
        return 1;
    }

    std::string prefix = argv[2];
    std::string binPath = prefix + ".bin";
    FILE* bin = fopen(binPath.c_str(), "wb");
    if (bin == nullptr)
    {
        printf("fopen(%s) failed: %s\n", binPath.c_str(), strerror(errno));
        return 1;
    }
    size_t written = 0;
    int err = 0;
    for (size_t off = 0; off < image.size; off += 0x10000)
    {
        size_t chunk = std::min<size_t>(0x10000, image.size - off);
        size_t w = fwrite(image.data.get() + off, 1, chunk, bin);
        written += w;
        if (w != chunk)
        {
            err = errno;
            printf("chunk at 0x%zX failed\n", off);
            break;
        }
    }
    fclose(bin);
    printf("wrote 0x%zX of 0x%X bytes (data=%p, errno=%d %s)\n", written, image.size, (void*)image.data.get(), err, strerror(err));

    FILE* f = fopen((prefix + ".txt").c_str(), "w");
    fprintf(f, "base 0x%zX\nsize 0x%X\nentry 0x%zX\n", image.base, image.size, image.entry_point);
    for (auto& s : image.sections)
        fprintf(f, "section %s 0x%zX 0x%X flags %d\n", s.name.c_str(), s.base, s.size, (int)s.flags);
    for (auto& sym : image.symbols)
        fprintf(f, "symbol 0x%zX 0x%zX %d %s\n", sym.address, sym.size, (int)sym.type, sym.name.c_str());
    fclose(f);

    // Full disassembly listing of every code section.
    FILE* a = fopen((prefix + ".asm").c_str(), "w");
    for (auto& s : image.sections)
    {
        if (!(s.flags & SectionFlags_Code))
            continue;
        fprintf(a, "; section %s\n", s.name.c_str());
        for (size_t off = 0; off + 4 <= s.size; off += 4)
        {
            ppc_insn insn;
            const uint8_t* p = s.data + off;
            uint32_t raw = (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
            ppc::Disassemble(p, s.base + off, insn);
            if (insn.opcode == nullptr)
                fprintf(a, "%08zX %08X .long 0x%08X\n", s.base + off, raw, raw);
            else
                fprintf(a, "%08zX %08X %s %s\n", s.base + off, raw, insn.opcode->name, insn.op_str);
        }
    }
    fclose(a);
    printf("ok\n");
    return 0;
}
