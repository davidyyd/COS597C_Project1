"""
static_check.py -- what a model file may contain beyond the model itself: nothing that reads or writes
files, opens connections, runs processes or loads outside weights, and nothing that touches gradients
(hooks, .grad, requires_grad), since how parameters are optimized is fixed by the harness. Checked
from the source (before the file is imported) by smoke.py, so ar-check and ar-exp reject such a model
with a VIOLATION line. (All 1,234 model files of the 2026-10-02 runs import only torch.)

Precision is the harness's too: a model file must be dtype-agnostic. Parameters, buffers and the
model's own arithmetic are fp32; the harness's bf16 autocast alone decides which ops run in lower
precision. So no lower-precision dtypes, no casts that follow another tensor's dtype (`.type_as`,
`.to(x.dtype)`, `.to(other)`), no dtype probing (`.dtype`, `element_size`), no autocast or global
numeric settings (TF32, default dtype), and no kernels or compiler settings of the model's own
(custom ops, `torch.ops`, `torch.compile`, inductor/dynamo config). Upcasts (`.float()`,
`.to(torch.float32)`) are allowed. precision_check.py checks the same at run time.
"""

import ast

ALLOWED_MODULES = {"torch", "math", "functools", "itertools", "typing", "dataclasses", "collections", "numbers",
                   "enum", "abc", "operator", "contextlib", "warnings", "einops"}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "breakpoint",
                   "input", "__builtins__", "__loader__", "__spec__"}
FORBIDDEN_ATTRS = {"load", "save", "hub", "from_pretrained", "load_state_dict_from_url", "load_library",
                   "load_inline", "cpp_extension", "from_file", "system", "popen", "__subclasses__", "__globals__",
                   "__builtins__", "__code__", "__import__", "_C", "__dict__", "__getattribute__",
                   # gradient manipulation: the optimizer is the harness's; a model may not touch gradients
                   "grad", "register_hook", "register_backward_hook", "register_full_backward_hook",
                   "register_full_backward_pre_hook", "register_post_accumulate_grad_hook",
                   "register_multi_grad_hook", "requires_grad", "requires_grad_", "retain_grad"}
PRECISION = "(precision is the harness's: model files must be dtype-agnostic)"
# lower-precision and quantized dtypes: not allowed as `torch.X` or as a cast method `.X()`
LOW_DTYPES = {"bfloat16", "float16", "half", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz",
              "float8_e8m0fnu", "float4_e2m1fn_x2", "int8", "uint8", "int4", "uint4", "qint8", "quint8", "qint32",
              "quint4x2", "complex32", "chalf"}
# never allowed: casts that follow another tensor's dtype, dtype probing, low-precision kernels
PRECISION_ATTRS = {"type_as", "dtype", "element_size", "itemsize", "_scaled_mm", "_int_mm", "_weight_int8pack_mm",
                   "_weight_int4pack_mm", "_convert_weight_to_int4pack"}
# not allowed on a module (torch.X, torch.backends.X, F.X, ...): autocast, global numeric settings, custom
# kernels and ops, compiler settings
MODULE_ATTRS = {"autocast", "amp", "is_autocast_enabled", "is_autocast_available", "get_autocast_dtype",
                "get_autocast_gpu_dtype", "set_autocast_enabled", "set_autocast_dtype", "set_default_dtype",
                "get_default_dtype", "set_float32_matmul_precision", "get_float32_matmul_precision", "backends",
                "allow_tf32", "allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction",
                "quantize_per_tensor", "quantize_per_channel", "quantization", "ao", "library", "custom_op",
                "triton_op", "wrap_triton", "ops", "compile", "compiler", "_dynamo", "_inductor", "jit"}
# dtypes a model may name (as `torch.X` in `dtype=...` or `.to(torch.X)`): fp32 and wider, integers, bool
ALLOWED_DTYPES = {"float32", "float", "float64", "double", "int16", "short", "int32", "int", "int64", "long", "bool"}


def _module_names(tree):
    """names bound to imported modules (torch, F, nn, math, ...)"""
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            names |= {a.asname or a.name for a in n.names}
    return names


def _root(node):
    """the name an attribute chain starts from (torch.backends.cuda -> torch)"""
    while isinstance(node, (ast.Attribute, ast.Call, ast.Subscript)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_allowed_dtype(node):
    return isinstance(node, ast.Attribute) and node.attr in ALLOWED_DTYPES


def _to_violation(call):
    """`.to(...)` may move a tensor to a device or upcast it to an allowed dtype, nothing else"""
    for a in call.args:
        ok = (_is_allowed_dtype(a) or (isinstance(a, ast.Constant) and isinstance(a.value, str))
              or (isinstance(a, ast.Attribute) and a.attr == "device"))
        if not ok:
            return (f"`.to({ast.unparse(a)})`: `.to` may only take a device or one of torch."
                    f"{'/'.join(sorted(ALLOWED_DTYPES))} {PRECISION}")
    return None


def violations(source):
    """list of reasons this model source is not allowed (empty if it is)"""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    modules = _module_names(tree)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mods = [n.module or ""] if n.level == 0 else ["<relative import>"]
            for a in n.names:     # `from torch import bfloat16` etc.
                if a.name in PRECISION_ATTRS | MODULE_ATTRS | LOW_DTYPES:
                    out.append(f"line {n.lineno}: import of `{a.name}` {PRECISION}")
        else:
            mods = []
        for m in mods:
            if m.split(".")[0] not in ALLOWED_MODULES:
                out.append(f"line {n.lineno}: import of `{m}` (model files may import only "
                           f"{', '.join(sorted(ALLOWED_MODULES))})")
            elif any(part in PRECISION_ATTRS | MODULE_ATTRS for part in m.split(".")[1:]):
                out.append(f"line {n.lineno}: import of `{m}` {PRECISION}")
        if isinstance(n, ast.Name) and n.id in FORBIDDEN_NAMES:
            out.append(f"line {n.lineno}: use of `{n.id}`")
        if isinstance(n, ast.Attribute) and n.attr in FORBIDDEN_ATTRS:
            out.append(f"line {n.lineno}: use of `.{n.attr}`")
        if isinstance(n, ast.Attribute):
            on_module = _root(n) in modules
            if n.attr in PRECISION_ATTRS or (on_module and (n.attr in MODULE_ATTRS or n.attr in LOW_DTYPES)):
                out.append(f"line {n.lineno}: use of `{ast.unparse(n)}` {PRECISION}")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and (
                n.func.attr in LOW_DTYPES or n.func.attr == "type"):
            out.append(f"line {n.lineno}: cast `.{n.func.attr}(...)` {PRECISION}")
        if isinstance(n, ast.keyword) and n.arg == "dtype" and not _is_allowed_dtype(n.value):
            out.append(f"line {n.value.lineno}: `dtype={ast.unparse(n.value)}`: only torch."
                       f"{'/'.join(sorted(ALLOWED_DTYPES))} {PRECISION}")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "to":
            v = _to_violation(n)
            if v:
                out.append(f"line {n.lineno}: {v}")
        # getattr / setattr / hasattr: a constant name is checked like an attribute; a computed name
        # on a module (getattr(torch, "bf" + "loat16")) is not allowed
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("getattr", "setattr", "hasattr", "delattr") \
                and len(n.args) >= 2:
            obj, name = n.args[0], n.args[1]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                if name.value in FORBIDDEN_ATTRS | PRECISION_ATTRS | MODULE_ATTRS | LOW_DTYPES:
                    out.append(f"line {n.lineno}: use of `.{name.value}` through {n.func.id}")
            elif isinstance(obj, ast.Name) and obj.id in modules or isinstance(obj, ast.Attribute):
                out.append(f"line {n.lineno}: {n.func.id} with a computed name on `{ast.unparse(obj)}`")
    return sorted(set(out), key=lambda s: int(s.split()[1].rstrip(":")) if s.startswith("line") else 0)
