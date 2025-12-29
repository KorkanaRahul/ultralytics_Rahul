import os
import torch
# import onnx

# ---- FIX: Dummy registry for unknown custom classes ----
# Prevents torch.load() from failing on missing layers/losses
import torch.nn as nn

class Dummy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

# Register a universal dummy class catch-all
globals()['DFLoss'] = Dummy
globals()['ECA'] = Dummy
globals()['WeightedAdd'] = Dummy
globals()['GhostC2fskip'] = Dummy
globals()['SPPF'] = Dummy
globals()['DWConv'] = Dummy
globals()['CBAM'] = Dummy
globals()['PSA'] = Dummy

# ---------------------------------------------------------

def inspect_model(file):
    print("="*10, f"Inspecting {os.path.basename(file)}", "="*10)

    ext = file.split(".")[-1].lower()

    # --------------------- ONNX ---------------------
    if ext == "onnx":
        print("Detected ONNX model")
        model = onnx.load(file)
        print(f"IR version: {model.ir_version}")

        print("\nInputs:")
        for i in model.graph.input:
            shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
            print(f" - {i.name}: {shape}")

        print("\nOutputs:")
        for o in model.graph.output:
            shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
            print(f" - {o.name}: {shape}")

        print("\nMetadata props:")
        if len(model.metadata_props) == 0:
            print(" - None")
        else:
            for p in model.metadata_props:
                print(f" - {p.key}: {p.value}")
        return

    # ----------------------- PT ----------------------
    if ext in ["pt", "pth"]:
        print("Detected PyTorch checkpoint (.pt)")

        try:
            ckpt = torch.load(file, map_location="cpu", weights_only=False)
            print("Checkpoint loaded successfully")

            # 1. Try Ultralytics format
            if isinstance(ckpt, dict):
                print("\nKeys:", list(ckpt.keys()))

                # ---- Class names ----
                names = None
                if "names" in ckpt:
                    names = ckpt["names"]
                elif "model" in ckpt and hasattr(ckpt["model"], "names"):
                    names = ckpt["model"].names

                if names:
                    print(f"\nClasses ({len(names)}):")
                    print(names)
                else:
                    print("\nClasses: Not found")

                # ---- nc ----
                if "nc" in ckpt:
                    print(f"\nnc: {ckpt['nc']}")

                # ---- Training args ----
                print("\nMetadata / Train args:")
                for key in ["args", "train_args", "hyperparams", "hyp"]:
                    if key in ckpt:
                        print(f"- {key}: {ckpt[key]}")

                # ---- Model structure ----
                if "model" in ckpt:
                    print("\nModel type:", type(ckpt["model"]))
                    try:
                        print("Total modules:", len(list(ckpt["model"].modules())))
                    except:
                        pass

            else:
                print("Unknown checkpoint structure")

        except Exception as e:
            print("❌ Failed to load checkpoint:", e)

    else:
        print("❌ Unsupported format:", ext)

# ---- PATCH FOR MISSING CUSTOM YOLO LOSS ----
import torch.nn as nn
# import ultralytics.utils.loss as loss_module

# class DFLoss(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#     def forward(self, *args, **kwargs):
#         return 0

# setattr(loss_module, "DFLoss", DFLoss)
# print("✓ Patched ultralytics.utils.loss.DFLoss")

# ---------- USAGE ----------
# inspect_model("best.pt")

folder = "./models/"
for filename in os.listdir(folder):
    if filename.endswith(".pt"):
        inspect_model(os.path.join(folder, filename))