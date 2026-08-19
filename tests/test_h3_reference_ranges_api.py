from types import SimpleNamespace

from flux_untwist.patches import minimax_h3_visual_reference_ranges


def _payload():
    layout = SimpleNamespace(
        segments=[
            (0, 10, "text"),
            (10, 18, "ref_img"),
            (18, 26, "ref_img"),
            (26, 40, "audio"),
            (40, 80, "video"),
        ]
    )
    refs = [{"kind": "image"}, {"kind": "video"}]
    return {"layout": layout, "refs": refs}


def test_h3_reference_ranges_accept_complete_payload():
    payload = _payload()
    assert minimax_h3_visual_reference_ranges(
        payload,
        scope="image_and_video",
    ) == [(10, 18), (18, 26)]


def test_h3_reference_ranges_accept_layout_and_refs_arguments():
    payload = _payload()
    assert minimax_h3_visual_reference_ranges(
        payload["layout"],
        payload["refs"],
        scope="image_and_video",
    ) == [(10, 18), (18, 26)]
