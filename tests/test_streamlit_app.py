from streamlit_app import parse_panel


def test_parse_panel_keeps_only_valid_values_and_accepts_decimal_comma() -> None:
    panel, invalid_fields = parse_panel(
        {"LBXSATSI": "12,5", "LBDSALSI": "", "LBXSAPSI": "not-a-number"}
    )

    assert panel == {"LBXSATSI": 12.5}
    assert invalid_fields == ["LBXSAPSI"]
