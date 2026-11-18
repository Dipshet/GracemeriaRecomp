RenoDX shader library (https://github.com/clshortfuse/renodx, MIT, see LICENSE), src/shaders at commit
a99ff41c3e0d43030a65442e0800a199234a8417, unmodified. Used by the runtime's HDR output pass: RenoDRT
(tonemap/reno_drt.hlsl), PsychoV-30 (tonemap/psychov/test30.hlsl) and the inverse tone mapping (inverse_tonemap.hlsl).
Only those and the files they include are kept (the other tone mappers, ACES among them, removed).
