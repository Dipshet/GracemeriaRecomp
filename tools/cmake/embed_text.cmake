# Embeds a text file as a C++ string constant: cmake -DINPUT=... -DOUTPUT=... -DSYMBOL=... -P embed_text.cmake
file(READ "${INPUT}" CONTENT)
file(WRITE "${OUTPUT}" "// Generated from ${INPUT}. Do not edit.\nnamespace ac6::gpu {\nextern const char* const ${SYMBOL} = R\"__ac6embed__(${CONTENT})__ac6embed__\";\n}\n")
