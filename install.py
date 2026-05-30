import launch

if not launch.is_installed("gradio"):
    try:
        launch.run_pip("install gradio", "requirements for nai multi subject")
    except Exception:
        pass
