"""Conservative local corroboration for model-proposed blank answer regions."""


def is_local_blank(response):
    return bool(
        response
        and response.get("extraction_source") == "local_layout_blank"
        and response.get("blank_confirmed")
        and not str(response.get("raw_text") or "").strip()
    )


def blank_pixels(image, reference=None):
    """Reject any unexplained ink; this is evidence, never an OCR confidence score."""
    import numpy as np

    pixels = np.asarray(image.convert("RGB")).min(axis=2).astype(np.int16)
    if min(pixels.shape) < 16:
        return {"confirmed": False, "reason": "区域分辨率不足"}
    background = float(np.percentile(pixels, 90))
    if background < 220:
        return {"confirmed": False, "reason": "背景过暗，无法确认浅色笔迹"}
    ink = (pixels < 253) & (pixels <= background - 3)
    if reference is not None:
        ref = np.asarray(reference.convert("RGB")).min(axis=2).astype(np.int16)
        if ref.shape != pixels.shape:
            return {"confirmed": False, "reason": "空白参考区域尺寸不匹配"}
        # No dilation: a small student mark near a printed line must remain visible.
        residual = ink & (ref - pixels >= 3)
        method = "blank_reference_difference"
    else:
        residual = ink.copy()
        h, w = pixels.shape
        bounds = [0, 0, w, h]
        # Only near-edge, full-width/full-height thin rules may be ignored.
        for axis, length, edge in [(1, h, max(3, round(h * 0.2))), (0, w, max(3, round(w * 0.2)))]:
            indices = np.flatnonzero(ink.mean(axis=axis) >= 0.95)
            border_groups = []
            for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
                if not len(group) or len(group) > max(2, round(length * 0.04)):
                    continue
                if group[-1] >= edge and group[0] < length - edge:
                    continue
                border_groups.append(group)
                if axis == 1:
                    residual[group, :] = False
                else:
                    residual[:, group] = False
            top = [g for g in border_groups if g[-1] < edge]
            bottom = [g for g in border_groups if g[0] >= length - edge]
            # A complete pair of cell borders identifies clipped neighbouring-row text.
            if top and bottom:
                lo, hi = int(top[-1][-1]) + 1, int(bottom[0][0])
                if axis == 1:
                    residual[:lo, :] = False
                    residual[hi:, :] = False
                    bounds[1], bounds[3] = lo, hi
                else:
                    residual[:, :lo] = False
                    residual[:, hi:] = False
                    bounds[0], bounds[2] = lo, hi
        method = "clear_region_with_edge_rules"
    count = int(residual.sum())
    return {
        "confirmed": count == 0,
        "method": method,
        "unexplained_ink_pixels": count,
        **({"evaluated_box": bounds} if reference is None else {}),
        "reason": "" if count == 0 else "仍有笔迹、印刷文字、阴影或噪声，不能确认为空白",
    }


def verify_blank_part(part, paths, references=None):
    if not part.get("blank_candidate"):
        return
    # Two explicit, complete blank observations are adopted automatically.
    # This is machine provenance, never teacher confirmation or a calibrated probability.
    part["blank_confirmed"] = part.get("consistent") is True
    part["blank_evidence"] = [
        {"method": "two_explicit_local_blank_observations", "confirmed": part["blank_confirmed"]}
    ]
    if not part["blank_confirmed"]:
        part["consistent"] = False
        part["issues"] = [*part.get("issues", []), "空白候选未通过局部笔迹核验，保留缺分"]
