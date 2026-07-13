import sys

import trident.data.copernicus.cmems as cmems


def test_credentials_status_does_not_use_removed_function(monkeypatch):
    class FakeClient:
        __version__ = "2.4.1"

        @staticmethod
        def login(**kwargs):
            return True

    monkeypatch.setattr(cmems, "is_available", lambda: (True, "installed"))
    monkeypatch.setattr(
        cmems,
        "_credentials_are_configured",
        lambda **kwargs: (True, "test"),
    )
    monkeypatch.setitem(sys.modules, "copernicusmarine", FakeClient)

    result = cmems.credentials_status(validate=True)

    assert result["configured"] is True
    assert result["valid"] is True
    assert "error" not in result


def test_unconfigured_credentials_are_reported_without_prompt(monkeypatch):
    monkeypatch.setattr(cmems, "is_available", lambda: (True, "installed"))
    monkeypatch.setattr(
        cmems,
        "_credentials_are_configured",
        lambda **kwargs: (False, None),
    )

    result = cmems.credentials_status()

    assert result["configured"] is False
    assert result["valid"] is False
