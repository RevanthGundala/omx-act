from deploy.modal_call_watchdog import record_value


def test_record_value_supports_modal_cli_json_formats():
    assert record_value({"App ID": "ap-title"}, "App ID") == "ap-title"
    assert record_value({"app_id": "ap-lower"}, "App ID") == "ap-lower"
    assert record_value({"Container ID": "ta-title"}, "Container ID") == "ta-title"
    assert record_value({"container_id": "ta-lower"}, "Container ID") == "ta-lower"
    assert record_value({"Filename": "checkpoints/080000"}, "Filename") == "checkpoints/080000"
    assert record_value({"filename": "checkpoints/080000"}, "Filename") == "checkpoints/080000"
