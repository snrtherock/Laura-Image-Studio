"""
Laura Image Studio - Workflow Health Check
Diagnostic node for verifying installation health, GPU status, and module availability.
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

OPTIONAL_DEPS = {
    "insightface": "face features",
    "onnxruntime": "face features",
    "cv2": "video/image processing",
    "scipy": "advanced processing",
}


class LauraWorkflowHealthCheck:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("STRING", {"default": "run"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("health_status", "health_report", "module_count")
    FUNCTION = "check"
    CATEGORY = "Laura Studio/Core"
    DESCRIPTION = "Check the health of the Laura Image Studio installation"

    def check(self, trigger):
        lines = ["=== Laura Studio Health Check ==="]
        issues = []

        # --- Import health ---
        import_health = {}
        try:
            from custom_nodes.Laura_Image_Studio import get_import_health
            import_health = get_import_health()
        except Exception:
            try:
                import importlib
                pkg = importlib.import_module("custom_nodes.Laura_Image_Studio")
                import_health = getattr(pkg, "get_import_health", lambda: {})()
            except Exception:
                issues.append("Could not retrieve import health")

        loaded = sum(1 for v in import_health.values() if v == "ok")
        total = len(import_health) if import_health else 0
        failed_modules = [k for k, v in import_health.items() if v not in ("ok", "disabled")]

        # --- GPU ---
        gpu_info = "Not available"
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                name = torch.cuda.get_device_name(0)
                gpu_info = f"{name} (CUDA available)"
            else:
                gpu_info = "No CUDA device detected"
        except Exception:
            gpu_info = "torch not available"

        # --- Model registry ---
        registry_version = "unknown"
        model_count = 0
        non_commercial_models = []
        try:
            from custom_nodes.Laura_Image_Studio.nodes import model_registry as mr
            registry_version = mr.get_registry_version()
            registry = getattr(mr, "MODEL_REGISTRY", {})
            model_count = len(registry)
            for model_id, info in registry.items():
                lic = info.get("license", "")
                if "Non-Commercial" in lic or "Non-commercial" in lic:
                    display = info.get("name", model_id)
                    non_commercial_models.append(display)
        except Exception as exc:
            issues.append(f"Model registry error: {exc}")

        # --- Core dependencies ---
        core_deps = ["torch", "PIL", "numpy", "requests"]
        dep_status = {}
        for dep in core_deps:
            try:
                __import__(dep)
                dep_status[dep] = True
            except ImportError:
                dep_status[dep] = False

        opt_dep_status = {}
        for dep, desc in OPTIONAL_DEPS.items():
            try:
                __import__(dep)
                opt_dep_status[dep] = True
            except ImportError:
                opt_dep_status[dep] = False

        # --- Determine status ---
        core_missing = [d for d in core_deps if not dep_status[d]]
        has_torch = dep_status.get("torch", False)

        if not has_torch or failed_modules:
            is_critical = not has_torch or any(
                m in failed_modules for m in ("model_registry", "model_manager", "generation", "control_center")
            )
            if is_critical:
                status = "critical"
            else:
                status = "degraded"
        elif core_missing:
            status = "degraded"
        elif any(not v for v in opt_dep_status.values()):
            status = "degraded" if loaded < total else "healthy"
        else:
            status = "healthy"

        # --- Build report ---
        lines.append(f"Status: {status.upper()}")
        lines.append("")
        lines.append(f"Modules: {loaded}/{total} loaded")
        lines.append(f"Registry: v{registry_version} ({model_count} models)")
        lines.append(f"GPU: {gpu_info}")
        lines.append("")

        dep_line_parts = []
        for dep in core_deps:
            mark = "✓" if dep_status[dep] else "✗"
            dep_line_parts.append(f"{mark} {dep}")
        lines.append("Dependencies:")
        while dep_line_parts:
            chunk, dep_line_parts = dep_line_parts[:4], dep_line_parts[4:]
            lines.append("  " + "    ".join(chunk))

        for dep, desc in OPTIONAL_DEPS.items():
            if not opt_dep_status.get(dep, False):
                lines.append(f"  ✗ {dep} (optional - {desc})")

        lines.append("")
        if failed_modules:
            lines.append(f"Failed Modules: {', '.join(failed_modules)}")
        else:
            lines.append("Failed Modules: none")

        if non_commercial_models:
            lines.append("")
            lines.append("License Notice:")
            for m in non_commercial_models:
                lines.append(f"  ! {m} - Non-Commercial license")

        if issues:
            lines.append("")
            lines.append("Warnings:")
            for issue in issues:
                lines.append(f"  - {issue}")

        report = "\n".join(lines)
        return (status, report, loaded)


NODE_CLASS_MAPPINGS["LauraWorkflowHealthCheck"] = LauraWorkflowHealthCheck
NODE_DISPLAY_NAME_MAPPINGS["LauraWorkflowHealthCheck"] = "Laura Workflow Health Check"
