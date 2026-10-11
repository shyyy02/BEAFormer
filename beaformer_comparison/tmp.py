"""Profile segmentation models in eval mode, with one RGB 224x224 image.

Install in your existing model environment: python -m pip install fvcore
Run from repository root:
    python beaformer_comparison/tmp.py --gpu 0
    python beaformer_comparison/tmp.py --models unet swin_umamba beaformer --gpu 0
    python beaformer_comparison/tmp.py --models all
    python beaformer_comparison/tmp.py --models unet --device cpu

GMACs counts supported conv/linear/matrix contractions only.
GFLOPs_fma1 counts all fvcore-supported operations, with one fused multiply-add
counted as one operation. Do not divide this total by two to obtain MACs.
Unsupported operators are flagged and saved. No checkpoint/config is required.
Default: the 14 comparison models plus BEAFormer. --models all also includes
other registered models. Swin_umamba requires the project's Mamba/CUDA dependencies;
its selective-scan custom operator may be unsupported and must be reviewed.
All operations actually executed by eval forward are included where supported,
including auxiliary outputs computed by the original model implementation.
"""

import argparse
from contextlib import contextmanager
import csv
import gc
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import types

import torch


# Same constructors as models/model_lib.py; import only the selected models.
MODEL_SPECS = {
    "unet": ("U_Net", "U_Net"),
    "uctransnet": ("UCTransNet", "UCTransNet"),
    "swinunet": ("SwinUnet", "swinunet"),
    "swin_umamba": ("Swin_umamba", "swin_umamba"),
    "pvtemcad": ("PVT_EMCAD", "EMCADNet"),
    "pranet": ("PraNet", "PraNet"),
    "beaformer": ("BEAFormer", "BEAFormer"),
    "pmrnet": ("PMRNet", "PMRNet"),
    "transunet": ("TransUNet", "TransUnet"),
    "lgmsnet": ("LGMSNet", "lgmsnet"),
    "cfformer": ("CFFormer", "cfformer"),
    "h2former": ("H2Former", "h2former"),
    "fcbformer": ("FCBFormer", "fcbformer"),
    "medt": ("MedT", "medt"),
    "ca_net": ("CA_Net", "ca_net"),
    "uacanet_l": ("UACANet", "UACANet"),
    "perspective_unet": ("Perspective_Unet", "perspective_unet"),
    "rollingunet_l": ("RollingUnet", "Rolling_Unet_L"),
    "mobileuvit_l": ("MobileUViT", "mobileuvit_l"),
    "utanet": ("UTANet", "UTANet"),
    "u_kan": ("U_KAN", "UKAN"),
    "missformer": ("MISSFormer", "MISSFormer"),
}
DEFAULT_MODELS = [
    "unet", "pranet", "uacanet_l", "swinunet", "uctransnet", "missformer",
    "h2former", "rollingunet_l", "swin_umamba", "pvtemcad", "u_kan",
    "mobileuvit_l", "cfformer", "utanet", "beaformer",
]
PRETRAINED_ARGUMENT = {
    "swinunet", "pvtemcad", "pranet", "transunet", "cfformer",
    "fcbformer", "uacanet_l", "utanet", "missformer", "swin_umamba",
}
MAC_OPERATORS = {"conv", "linear", "matmul", "bmm", "mm", "addmm", "einsum"}


@contextmanager
def model_namespace(directory):
    """Resolve models.* inside the intended project during construction/forward."""
    saved = {k: v for k, v in sys.modules.items()
             if k == "models" or k.startswith("models.")}
    for key in saved:
        del sys.modules[key]
    package = types.ModuleType("models")
    package.__path__ = [str(directory)]
    sys.modules["models"] = package
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key == "models" or key.startswith("models."):
                del sys.modules[key]
        sys.modules.update(saved)


def profile_model(name, device, num_classes, encoder):
    from fvcore.nn import FlopCountAnalysis

    project = Path(__file__).resolve().parent
    directory = (project.parent / "beaformer_ablation" / "models"
                 if name == "beaformer" else project / "models")
    module_name, constructor = MODEL_SPECS[name]
    kwargs = {"num_classes": num_classes}
    if name in PRETRAINED_ARGUMENT:
        kwargs["pretrained_backbone_fpath"] = None
    if name == "pranet":
        if num_classes != 1:
            raise ValueError("This project's PraNet only supports binary segmentation")
        kwargs.pop("num_classes")
    if name == "utanet":
        kwargs["n_classes"] = kwargs.pop("num_classes")
    if name == "beaformer":
        kwargs["encoder"] = encoder
    if name == "u_kan":
        kwargs["img_size"] = 224
    with model_namespace(directory):
        module = importlib.import_module(f"models.{module_name}")
        model = getattr(module, constructor)(**kwargs).to(device).eval()
        parameters = sum(p.numel() for p in model.parameters())
        image = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            analysis = FlopCountAnalysis(model, (image,))
            total = float(analysis.total())
            operators = {k: float(v) for k, v in analysis.by_operator().items()}
            unsupported = dict(analysis.unsupported_ops())
            uncalled = sorted(analysis.uncalled_modules())
        macs = sum(v for k, v in operators.items() if k in MAC_OPERATORS)
        return {"model": name,
                "status": "review_unsupported_ops" if unsupported else "ok_supported_ops",
                "Params_M": parameters / 1e6, "GMACs": macs / 1e9,
                "GFLOPs_fma1": total / 1e9, "parameters": parameters,
                "macs": macs, "fvcore_flops_fma1": total,
                "by_operator": operators, "unsupported_ops": unsupported,
                "uncalled_modules": uncalled, "error": ""}


def save_results(output, results, args, device):
    fields = ["model", "status", "Params_M", "GMACs", "GFLOPs_fma1", "error"]
    with open(output / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "input_shape": [1, 3, 224, 224], "mode": "eval", "device": str(device),
        "num_classes": args.num_classes, "encoder": args.encoder,
        "versions": {"torch": torch.__version__,
                     "fvcore": importlib.metadata.version("fvcore")},
        "MACs_definition": "Supported conv/linear/matmul/bmm/mm/addmm/einsum contractions only; excludes bias additions and elementwise routing operations",
        "FLOPs_definition": "fvcore supported operators; one fused multiply-add counts as one operation; includes supported non-MAC operations",
        "coverage": "Unsupported operators are not counted. Empty unsupported_ops does not mean every scalar operation is counted: fvcore ignores some operators by design.",
        "forward_scope": "Actual eval forward including auxiliary outputs if computed; no backward computation",
        "results": results,
    }
    with open(output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=[*MODEL_SPECS, "all"],
                        default=DEFAULT_MODELS)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--gpu", type=int, default=0, help="Index among visible CUDA devices")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--encoder", default="pvt_v2_b2", help="BEAFormer encoder")
    parser.add_argument("--output-dir", type=Path, default=Path("complexity_results"))
    args = parser.parse_args()
    if args.num_classes < 1:
        parser.error("--num-classes must be positive")
    try:
        import fvcore.nn
    except ImportError:
        parser.error("Install fvcore first: python -m pip install fvcore")
    if args.device == "cuda":
        if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
            parser.error("Select an available --gpu, or use --device cpu")
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if args.device == "cuda" else "cpu")
    names = list(MODEL_SPECS) if "all" in args.models else list(dict.fromkeys(args.models))
    torch.manual_seed(1234)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"Input: (1, 3, 224, 224); eval mode; device={device}", flush=True)
    print("GFLOPs_fma1: one fused multiply-add counts as one operation.", flush=True)
    for name in names:
        print(f"\nProfiling {name} ...", flush=True)
        try:
            result = profile_model(name, device, args.num_classes, args.encoder)
            print(f"{name}: Params={result['Params_M']:.4f} M; "
                  f"MACs={result['GMACs']:.4f} G; "
                  f"fvcore FLOPs (FMA=1)={result['GFLOPs_fma1']:.4f} G; "
                  f"status={result['status']}", flush=True)
            if result["unsupported_ops"]:
                print(f"Uncounted operators: {result['unsupported_ops']}", flush=True)
        except Exception as exc:
            result = {"model": name, "status": "failed", "Params_M": None,
                      "GMACs": None, "GFLOPs_fma1": None,
                      "error": f"{type(exc).__name__}: {exc}"}
            print(f"{name}: FAILED: {result['error']}", flush=True)
        results.append(result)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Save after each model, preserving partial results if interrupted.
        save_results(args.output_dir, results, args, device)
    print(f"\nSaved results to {args.output_dir.resolve()}", flush=True)
    if any(row["status"] == "failed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
