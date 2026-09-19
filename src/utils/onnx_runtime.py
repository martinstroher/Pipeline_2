"""Initialize local inference without ONNX Runtime telemetry workers."""


def configure_onnx_runtime() -> None:
    """Disable diagnostic telemetry before dependencies create inference sessions."""
    import onnxruntime

    onnxruntime.disable_telemetry_events()
