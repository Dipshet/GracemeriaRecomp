# Exports an IDA database's names as a symbols file for the Lua mods (ac6.sym, ac6.data, ac6.signatures:
# GracemeriaRecomp/src/mods/symbols.toml has the format). Run it in IDA on the game's default.xex (File > Script file...),
# then put the file in a symbols/ folder next to the game's executable (or in a mod's folder as symbols.toml).
#
# Exported: functions and data with names of their own (not IDA's sub_ / loc_ / unk_ / dword_ ...), and the signatures
# of functions whose IDA prototypes map onto the Lua API's types (void bool s8 ... u64 ptr f32 f64; at most 8 integer
# arguments). Names are written as they are in IDA; a name used twice keeps the first address.
import ida_bytes
import ida_funcs
import ida_kernwin
import ida_nalt
import ida_name
import ida_typeinf
import idautils
import idc

DEFAULT_PREFIXES = ("sub_", "nullsub_", "j_", "loc_", "locret_", "unk_", "byte_", "word_", "dword_", "qword_", "off_",
                    "flt_", "dbl_", "asc_", "stru_", "algn_", "jpt_", "def_", "unknown_libname", "__savegpr", "__restgpr",
                    "__savefpr", "__restfpr", "__savevmx", "__restvmx", "??_", "_TlsIndex")


def own_name(name):
    return name and not name.startswith(DEFAULT_PREFIXES)


def lua_type(t):
    if t.is_void():
        return "void"
    if t.is_ptr() or t.is_array() or t.is_func():
        return "ptr"
    if t.is_floating():
        return "f64" if t.get_size() == 8 else "f32"
    if t.is_bool():
        return "bool"
    if t.is_integral() or t.is_enum():
        size = t.get_size()
        table = {1: "8", 2: "16", 4: "32", 8: "64"}
        if size not in table:
            return None
        return ("s" if t.is_signed() else "u") + table[size]
    return None


def signature(ea):
    tif = ida_typeinf.tinfo_t()
    if not ida_nalt.get_tinfo(tif, ea):
        return None
    details = ida_typeinf.func_type_data_t()
    if not tif.get_func_details(details):
        return None
    ret = lua_type(details.rettype)
    args = [lua_type(a.type) for a in details]
    if ret is None or None in args:
        return None
    if sum(1 for a in args if a not in ("f32", "f64")) > 8:
        return None
    return "%s(%s)" % (ret, ", ".join(args))


def key(name):
    return '"%s"' % name.replace("\\", "\\\\").replace('"', '\\"')


def main():
    path = ida_kernwin.ask_file(1, "ida.toml", "Save the symbols file as")
    if not path:
        return
    functions, data, signatures = {}, {}, {}
    taken = set()
    for ea in idautils.Functions():
        name = ida_funcs.get_func_name(ea)
        if own_name(name) and name not in taken:
            taken.add(name)
            functions[name] = ea
            s = signature(ea)
            if s:
                signatures[name] = s
    for ea, name in idautils.Names():
        if ida_funcs.get_func(ea) is not None or not own_name(name) or name in taken:
            continue
        if not ida_bytes.is_data(ida_bytes.get_flags(ea)) and not ida_bytes.is_unknown(ida_bytes.get_flags(ea)):
            continue
        taken.add(name)
        data[name] = ea
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Names exported from IDA (tools/ida/export_symbols.py): %s\n\n" % idc.get_idb_path())
        f.write("[functions]\n")
        for name, ea in sorted(functions.items(), key=lambda x: x[1]):
            f.write("%s = 0x%08X\n" % (key(name), ea))
        f.write("\n[data]\n")
        for name, ea in sorted(data.items(), key=lambda x: x[1]):
            f.write("%s = 0x%08X\n" % (key(name), ea))
        f.write("\n[signatures]\n")
        for name, s in sorted(signatures.items()):
            f.write('%s = "%s"\n' % (key(name), s))
    print("export_symbols: %d functions, %d data, %d signatures -> %s" % (len(functions), len(data), len(signatures), path))


main()
