import os
import json
import hashlib
import importlib
import casadi as ca
from .cusadi_function import CusadiFunction
from .kernel_codegen import (
    build_kernel,
    build_pybind,
    get_codegen_kernel_names
)
from .utils.jit_kernels import compile_and_load_kernels

# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODEGEN_DIR = os.path.join(CURRENT_DIR, "codegen")
FUNCTION_DIR = os.path.join(CURRENT_DIR, "casadi_fns")
MANIFEST_PATH = os.path.join(CODEGEN_DIR, "codegen_manifest.json")

def _load_codegen_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {}
    try:
        with open(MANIFEST_PATH, "r") as manifest_file:
            return json.load(manifest_file)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_codegen_manifest(manifest):
    with open(MANIFEST_PATH, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)


def _fingerprint_function(fn):
    try:
        serialized = fn.serialize()
    except Exception:
        serialized = f"{fn.name()}::{fn.n_instructions()}::{fn.sz_w()}"
    if isinstance(serialized, str):
        serialized = serialized.encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _build_manifest_entry(fn, batch_size, precision, dynamic_batching):
    entry = {
        "name": fn.name(),
        "fingerprint": _fingerprint_function(fn),
        "n_in": fn.n_in(),
        "n_out": fn.n_out(),
        "n_instr": fn.n_instructions(),
        "sz_w": fn.sz_w(),
        "precision": precision,
        "dynamic_batching": dynamic_batching,
    }
    if not dynamic_batching:
        entry["batch_size"] = batch_size
    return entry


def _is_codegen_stale(fn, batch_size, precision, dynamic_batching, manifest):
    kernel_path = os.path.join(CODEGEN_DIR, f"{fn.name()}.cu")
    bindings_header_path = os.path.join(CODEGEN_DIR, "bindings.h")
    bindings_cpp_path = os.path.join(CODEGEN_DIR, "bindings.cpp")
    if not os.path.exists(kernel_path):
        return True
    if not os.path.exists(bindings_header_path) or not os.path.exists(bindings_cpp_path):
        return True
    with open(bindings_header_path, "r") as header_file:
        header_content = header_file.read()
    with open(bindings_cpp_path, "r") as cpp_file:
        cpp_content = cpp_file.read()
    if f"void {fn.name()}_binding(" not in header_content:
        return True
    if f'm.def("{fn.name()}", &{fn.name()}_binding);' not in cpp_content:
        return True
    expected = _build_manifest_entry(fn, batch_size, precision, dynamic_batching)
    current = manifest.get(fn.name())
    return current != expected


def _has_loaded_kernel_bindings(fn_names):
    try:
        module = importlib.import_module("cusadi_kernels")
    except ModuleNotFoundError:
        return False
    return all(hasattr(module, fn_name) for fn_name in fn_names)


def codegen_functions(fns,
                      batch_size=1,
                      precision='float',
                      dynamic_batching=True):
    casadi_fns = []
    for fn in fns:
        if isinstance(fn, str):
            fn_filepath = os.path.join(FUNCTION_DIR, f"{fn}.casadi")
            try:
                casadi_fn = ca.Function.load(fn_filepath)
            except Exception as e:
                print(f"Error loading {fn_filepath}: {e}")
        elif isinstance(fn, ca.Function):
            casadi_fn = fn
        casadi_fns.append(casadi_fn)
        print(f"Loaded CasADi function: {casadi_fn.name()} ({casadi_fn.n_instructions()} instructions)")

    manifest = _load_codegen_manifest()
    manifest_updated = False
    for fn in casadi_fns:
        if _is_codegen_stale(fn, batch_size, precision, dynamic_batching, manifest):
            build_kernel(fn, batch_size=batch_size,
                         precision=precision, dynamic_batching=dynamic_batching)
            build_pybind(fn, precision=precision)
            manifest[fn.name()] = _build_manifest_entry(
                fn, batch_size, precision, dynamic_batching
            )
            manifest_updated = True
        else:
            print(f"Skipping codegen for {fn.name()}; cached sources are up to date.")

    if manifest_updated:
        _save_codegen_manifest(manifest)

    return casadi_fns, manifest_updated


def parallelize_functions(fns,
                          batch_size=1,
                          precision='float',
                          dynamic_batching=True):
    casadi_fns, manifest_updated = codegen_functions(
        fns,
        batch_size=batch_size,
        precision=precision,
        dynamic_batching=dynamic_batching,
    )

    kernel_names = get_codegen_kernel_names()
    requested_names = [fn.name() for fn in casadi_fns]
    if manifest_updated or not _has_loaded_kernel_bindings(requested_names):
        compile_and_load_kernels(kernel_names)
    else:
        print("Using loaded cusadi_kernels module; skipping JIT load.")

    cusadi_fns = {}
    for fn in casadi_fns:
        cusadi_fns[fn.name()] = CusadiFunction(fn, batch_size, precision, dynamic_batching)
    return cusadi_fns
