"""Local X3D candidate analysis and a deterministic, no-network MoTrack agent.

This module deliberately treats Kinetics-400 labels as *action evidence*, not as
ground truth for tennis outcomes.  It is an extension of HypeCut's existing
signals: low-rate whole-video windows prevent HypeCut-only blind spots, while
dense windows around HypeCut segments add temporal evidence where it matters.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .security import session_directory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "lightweight"
def _media_binary(name: str) -> Path:
    """Find FFmpeg without hard-wiring this clone to one checkout layout.

    A packaged binary remains preferred.  ``MOTRACK_FFMPEG`` can point to an
    external FFmpeg directory or executable; the sibling legacy checkout is a
    local-development fallback while a fresh clone can use PATH.
    """
    configured = os.getenv("MOTRACK_FFMPEG", "").strip()
    candidates = [ROOT / "tools" / "ffmpeg" / "bin" / name]
    if configured:
        base = Path(configured).expanduser()
        candidates.append(base / name if base.is_dir() else base)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    # Keep a clear path in errors for installations that have not installed it.
    return candidates[0]


FFMPEG = _media_binary("ffmpeg")
FFPROBE = _media_binary("ffprobe")
TORCH_HOME = ROOT / ".cache" / "torch"
LABELS_PATH = ROOT / ".cache" / "lightweight" / "kinetics_classnames.json"
VIDEO_IDS: dict[str, Path] = {}
HYPECUT: dict[str, Path] = {}
HYPECUT_FALLBACK_DIRS: dict[str, Path] = {}
EVENT_TYPES = {"serve", "tennis_hit", "tennis_action_candidate", "UNKNOWN"}
ALIASES = {"发球": "serve", "serve": "serve", "击球": "tennis_hit", "回合": "tennis_hit", "tennis": "tennis_action_candidate"}
ANALYSIS_VERSION = "4.0"  # Tennis-only Kinetics action evidence; no score or point outcome inference.
SHOT_LIKE_TYPES = {"serve", "tennis_hit"}
TENNIS_LABELS = {
    "tennis serve": ("serve", "疑似发球"),
    "playing tennis": ("tennis_action_candidate", "网球动作"),
}
EVENT_TYPE_ZH = {
    "serve": "疑似发球",
    "tennis_hit": "疑似击球",
    "tennis_action_candidate": "网球动作",
    "hypecut_candidate": "HypeCut 信号片段",
    "UNKNOWN": "未知动作",
}

# ---------- editing language / renderer ----------
# The local agent is also a tiny deterministic editor.  These presets keep the
# result readable: a tight reel uses more cuts, a relaxed reel lets a play
# breathe.  They are intentionally exposed in the UI so users can see exactly
# what the machine is doing.
PACE_PRESETS = {
    "tight": {"segment": 2.6, "min": 1.6, "max": 3.4, "pre": 1.1, "post": 1.2, "max_clips": 14},
    "standard": {"segment": 3.9, "min": 2.6, "max": 5.2, "pre": 1.6, "post": 1.9, "max_clips": 10},
    "relaxed": {"segment": 5.8, "min": 4.0, "max": 7.6, "pre": 2.3, "post": 2.6, "max_clips": 7},
}
PACE_ZH = {"tight": "紧凑快剪", "standard": "标准节奏", "relaxed": "舒缓沉浸"}
TRANSITION_ZH = {"fade": "淡入淡出", "cut": "硬切直出", "flash": "闪白转场"}
REPLAY_SPEED = 0.55
REPLAY_MAX_SOURCE = 2.5
FONT_CJK = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
FONT_LATIN = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


LOOK_ZH = {"vivid": "鲜艳", "cinematic": "电影感", "retro": "复古", "bw": "黑白"}
STYLE_DEFAULTS = {"pace": "standard", "transition": "fade", "captions": True, "slow_motion": False,
                  "look": "vivid", "zoom": False, "music": "none", "title": None, "aspect": "16:9"}
ASPECTS = ("16:9", "9:16")


def _clean_title(value: Any) -> str | None:
    """None = default title; "" = no title; otherwise up to 20 safe characters."""
    if value is None:
        return None
    text = re.sub(r"[\x00-\x1f\x7f%\\{}]", "", str(value)).strip()
    return text[:20]


def normalize_edit_style(value: Any = None) -> dict[str, Any]:
    """Return a validated editing style, accepting dicts or short strings."""
    style = dict(STYLE_DEFAULTS)
    if isinstance(value, str):
        return edit_style_from_text(value, style)
    if isinstance(value, dict):
        pace = str(value.get("pace", style["pace"])).lower()
        if pace in PACE_PRESETS:
            style["pace"] = pace
        transition = str(value.get("transition", style["transition"])).lower()
        if transition in TRANSITION_ZH:
            style["transition"] = transition
        aspect = str(value.get("aspect", style["aspect"]))
        if aspect in ASPECTS:
            style["aspect"] = aspect
        look = str(value.get("look", style["look"])).lower()
        if look in LOOK_ZH:
            style["look"] = look
        for key in ("captions", "slow_motion", "zoom"):
            if key in value:
                raw = value[key]
                if isinstance(raw, str):
                    style[key] = raw.strip().lower() in {"1", "true", "yes", "on", "是"}
                else:
                    style[key] = bool(raw)
        if "music" in value:
            from .music import valid_choice
            style["music"] = value["music"] if valid_choice(value["music"]) else "none"
        if "title" in value:
            style["title"] = _clean_title(value["title"])
    return style


STYLE_WORDS = {
    "pace_tight": ("紧凑", "快剪", "快节奏", "利落", "节奏快"),
    "pace_standard": ("标准节奏", "正常节奏"),
    "pace_relaxed": ("舒缓", "慢节奏", "沉浸", "呼吸感", "节奏慢"),
    "cut": ("硬切", "直切", "不要转场", "无转场", "去掉转场"),
    "flash": ("闪白", "闪光", "白闪", "闪一下", "闪切"),
    "fade": ("淡入淡出", "黑场", "柔和转场", "转场"),
    "captions_off": ("不要字幕", "无字幕", "去掉字幕", "关闭字幕", "关掉字幕", "不加字幕"),
    "captions_on": ("字幕", "说明文字"),
    "slow_off": ("不要慢放", "关闭慢放", "去掉慢放", "关掉慢放", "不加慢放", "不要慢动作", "去掉慢动作", "不要回放", "去掉回放", "正常速度"),
    "slow_on": ("慢放", "慢动作", "回放", "slow motion"),
    "music_off": ("不要音乐", "去掉音乐", "关掉音乐", "不要配乐", "去掉配乐", "关掉配乐", "不要背景音乐", "去掉背景音乐", "只要原声", "用原声"),
    "music_hype": ("电子", "edm", "热血", "燃"),
    "music_groove": ("嘻哈", "说唱", "hiphop", "hip hop", "律动", "街头"),
    "music_chill": ("轻快", "轻松", "清爽", "欢快"),
    "music_on": ("配乐", "音乐", "bgm", "背景音", "伴奏", "换一首"),
    "look_retro": ("复古", "胶片", "老电影", "怀旧"),
    "look_cinematic": ("电影感", "电影风", "大片", "宽银幕"),
    "look_bw": ("黑白",),
    "look_vivid": ("鲜艳", "去掉滤镜", "不要滤镜", "原色", "正常颜色"),
    "aspect_h": ("横屏", "16:9", "宽屏", "不要竖屏", "取消竖屏"),
    "aspect_v": ("竖屏", "9:16", "抖音", "视频号", "快手", "小红书", "竖版", "手机竖"),
    "zoom_off": ("不要推镜", "去掉推镜", "关掉推镜", "不要缩放", "不要特效", "去掉特效", "关掉特效"),
    "zoom_on": ("推镜", "镜头推进", "缩放", "动感镜头", "运镜", "特效"),
}
TITLE_SET = re.compile(r"(?:标题|片头)(?:改成|改为|换成|叫|写|用|是)[“\"'「『]?([^”\"'」』，。,.!！?？\n]{1,20})")
TITLE_OFF = ("不要标题", "去掉标题", "关掉标题", "不要片头", "去掉片头")


def _has(low: str, key: str) -> bool:
    return any(word in low for word in STYLE_WORDS[key])


def edit_style_from_text(text: str, base: dict[str, Any] | None = None) -> dict[str, Any]:
    # Negations are checked first: "不要慢放" also contains "慢放".
    style = normalize_edit_style(base)
    raw = text or ""
    low = raw.lower()
    if _has(low, "pace_tight"):
        style["pace"] = "tight"
    elif _has(low, "pace_relaxed"):
        style["pace"] = "relaxed"
    elif _has(low, "pace_standard"):
        style["pace"] = "standard"
    if _has(low, "cut"):
        style["transition"] = "cut"
    elif _has(low, "flash"):
        style["transition"] = "flash"
    elif _has(low, "fade"):
        style["transition"] = "fade"
    if _has(low, "captions_off"):
        style["captions"] = False
    elif _has(low, "captions_on"):
        style["captions"] = True
    if _has(low, "slow_off"):
        style["slow_motion"] = False
    elif _has(low, "slow_on"):
        style["slow_motion"] = True
    # Music: genre words name a preset; mood words count only next to a music word.
    music_context = _has(low, "music_on")
    if _has(low, "music_off"):
        style["music"] = "none"
    elif any(w in low for w in ("电子", "edm")) or (music_context and _has(low, "music_hype")):
        style["music"] = "hype"
    elif any(w in low for w in ("嘻哈", "说唱", "hiphop", "hip hop")) or (music_context and _has(low, "music_groove")):
        style["music"] = "groove"
    elif music_context and _has(low, "music_chill"):
        style["music"] = "chill"
    elif "换一首" in low:
        order = ["hype", "groove", "chill"]
        current = style["music"] if style["music"] in order else order[-1]
        style["music"] = order[(order.index(current) + 1) % len(order)]
    elif music_context and style["music"] == "none":
        style["music"] = "hype"
    if _has(low, "look_retro"):
        style["look"] = "retro"
    elif _has(low, "look_cinematic"):
        style["look"] = "cinematic"
    elif _has(low, "look_bw"):
        style["look"] = "bw"
    elif _has(low, "look_vivid"):
        style["look"] = "vivid"
    if _has(low, "aspect_h"):
        style["aspect"] = "16:9"
    elif _has(low, "aspect_v"):
        style["aspect"] = "9:16"
    if _has(low, "zoom_off"):
        style["zoom"] = False
        if "特效" in low and style["transition"] == "flash":
            style["transition"] = "fade"
    elif _has(low, "zoom_on"):
        style["zoom"] = True
        if "特效" in low and not (_has(low, "cut") or _has(low, "fade")):
            style["transition"] = "flash"
    if any(w in low for w in TITLE_OFF):
        style["title"] = ""
    else:
        match = TITLE_SET.search(raw)
        if match:
            style["title"] = _clean_title(match.group(1))
    return style


def mentions_style(text: str) -> bool:
    low = (text or "").lower()
    return (any(_has(low, key) for key in STYLE_WORDS) or any(w in low for w in TITLE_OFF)
            or bool(TITLE_SET.search(text or "")))


def duration_meta(requested: float, source_duration: float) -> dict[str, Any]:
    """Explain the physical limit instead of silently returning the wrong length."""
    requested = max(0.0, float(requested))
    source_duration = max(0.0, float(source_duration))
    effective = min(requested, source_duration) if source_duration else requested
    effective = max(5.0, effective) if effective else effective
    clamped = source_duration > 0 and requested > source_duration + 0.05
    if clamped:
        note = (
            f"原片只有 {source_duration:.1f} 秒，无法生成 {requested:g} 秒；"
            f"已按全片时长生成 {effective:.1f} 秒的精华版。"
        )
    else:
        note = ""
    return {
        "requested_duration": round(requested, 3),
        "effective_duration": round(effective, 3),
        "source_duration": round(source_duration, 3),
        "duration_clamped": clamped,
        "duration_note": note,
    }


def _score_of(item: dict[str, Any]) -> float:
    for key in ("final_ranking_score", "raw_model_score", "ranking_score", "hypecut_score", "confidence"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _center_of(item: dict[str, Any]) -> float:
    return (float(item["start_time"]) + float(item["end_time"])) / 2.0


def _dedupe_events(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse near-duplicate scan windows, keeping the strongest evidence."""
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda x: (float(x["start_time"]), -_score_of(x))):
        if kept and float(item["start_time"]) <= float(kept[-1]["end_time"]) + 0.25:
            if _score_of(item) > _score_of(kept[-1]):
                kept[-1] = item
            continue
        kept.append(dict(item))
    return kept


def _select_for_montage(items: list[dict[str, Any]], count: int, min_gap: float = 1.6) -> list[dict[str, Any]]:
    if not items:
        return []
    unique = _dedupe_events(items)
    min_gap = max(1.6, float(min_gap))
    chosen: list[dict[str, Any]] = []
    for item in sorted(unique, key=_score_of, reverse=True):
        if any(abs(_center_of(item) - _center_of(other)) < min_gap for other in chosen):
            continue
        chosen.append(item)
        if len(chosen) >= count:
            break
    if not chosen:
        chosen = [max(unique, key=_score_of)]
    return sorted(chosen, key=lambda x: float(x["start_time"]))


def _waterfill(caps: list[float], target: float) -> list[float]:
    """Split ``target`` across clips without ever exceeding a clip's cap."""
    target = max(0.0, float(target))
    allocs = [0.0] * len(caps)
    remaining = list(range(len(caps)))
    left = target
    while remaining and left > 1e-6:
        share = left / len(remaining)
        capped = [i for i in remaining if caps[i] <= share + 1e-6]
        if not capped:
            for i in remaining:
                allocs[i] = share
            left = 0.0
            break
        for i in capped:
            allocs[i] = max(0.0, caps[i])
            left -= allocs[i]
            remaining.remove(i)
        left = max(0.0, left)
    # Floating-point residue belongs to the strongest (first) uncapped clip.
    diff = target - sum(allocs)
    if abs(diff) > 1e-9 and allocs:
        index = max(range(len(allocs)), key=lambda i: caps[i] - allocs[i])
        allocs[index] = max(0.0, min(caps[index], allocs[index] + diff))
    return allocs


def _expand_windows(windows: list[list[float]], limits: list[tuple[float, float]], target: float) -> None:
    """Grow context windows into unused source until they can hold ``target``."""
    deficit = target - sum(hi - lo for lo, hi in windows)
    if deficit <= 1e-6:
        return
    guard = 0
    while deficit > 1e-6 and guard < 40:
        guard += 1
        moved = 0.0
        for i, (lo, hi) in enumerate(windows):
            cap_left = max(0.0, lo - limits[i][0])
            cap_right = max(0.0, limits[i][1] - hi)
            capacity = cap_left + cap_right
            if capacity <= 1e-9:
                continue
            take = min(deficit - moved, capacity)
            if take <= 1e-9:
                continue
            total = cap_left + cap_right
            add_left = take * (cap_left / total) if total else 0.0
            add_right = take - add_left
            windows[i][0] = lo - add_left
            windows[i][1] = min(limits[i][1], hi + add_right)
            moved += take
        if moved <= 1e-6:
            break
        deficit -= moved


def _weighted_fill(caps: list[float], weights: list[float], target: float) -> list[float]:
    """Like _waterfill, but stronger plays get a larger share: the reel breathes
    instead of ticking at one fixed clip length."""
    allocs = [0.0] * len(caps)
    remaining = list(range(len(caps)))
    left = max(0.0, float(target))
    while remaining and left > 1e-6:
        total = sum(weights[i] for i in remaining) or 1.0
        shares = {i: left * weights[i] / total for i in remaining}
        capped = [i for i in remaining if caps[i] <= shares[i] + 1e-6]
        if not capped:
            for i in remaining:
                allocs[i] = shares[i]
            break
        for i in capped:
            allocs[i] = max(0.0, caps[i])
            left -= allocs[i]
            remaining.remove(i)
        left = max(0.0, left)
    return allocs


def camera_cuts(video_id: str) -> list[float]:
    """Camera-cut times in a source video (ffmpeg scene detection), cached."""
    video = VIDEO_IDS[video_id]
    path = OUT / f"shots_{video_id}.json"
    fingerprint = _fingerprint(video)
    cached = _json(path, {})
    if cached.get("fingerprint") == fingerprint:
        return list(cached.get("cuts", []))
    try:
        log = subprocess.run([str(FFMPEG), "-v", "info", "-i", str(video), "-an", "-vf",
                              "scale=320:-2,select='gt(scene,0.30)',showinfo", "-f", "null", "-"],
                             capture_output=True, text=True, timeout=300).stderr
    except (OSError, subprocess.SubprocessError):
        return []
    cuts = [round(float(x), 3) for x in re.findall(r"pts_time:([0-9.]+)", log)]
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fingerprint": fingerprint, "cuts": cuts}), encoding="utf-8")
    return cuts


def _merge_jump_cuts(segments: list[dict[str, Any]], shot_cuts: list[float] | None, source_duration: float) -> list[dict[str, Any]]:
    """Two clips a moment apart inside one camera shot read as a glitch, so join them."""
    if len(segments) < 2:
        return segments
    limit = 2.5 if shot_cuts is not None else 1.2
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = float(seg["start_time"]) - float(prev["end_time"])
        between = shot_cuts is not None and any(
            float(prev["end_time"]) - 0.3 <= c <= float(seg["start_time"]) + 0.3 for c in shot_cuts)
        if gap >= limit or between:
            merged.append(seg)
            continue
        out = float(prev["output_duration"]) + float(seg["output_duration"])
        mid = (float(prev["event_center"]) + float(seg["event_center"])) / 2.0
        start = min(max(0.0, mid - out / 2.0), max(0.0, source_duration - out))
        ids = prev.get("merged_event_ids") or [prev.get("event_id")]
        if _score_of(seg) > _score_of(prev):  # the stronger play names the clip
            for key in ("event_id", "event_type", "label", "display_name", "action_label_zh", "outcome", "raw_model_score"):
                prev[key] = seg.get(key)
        prev.update(start_time=round(start, 3), end_time=round(min(source_duration, start + out), 3),
                    output_duration=round(out, 3), event_center=round(mid, 3),
                    merged_event_ids=[*ids, seg.get("event_id")])
    return merged


def _avoid_slivers(segments: list[dict[str, Any]], cuts: list[float], source_duration: float, sliver: float = 0.7) -> None:
    """Shift a window so it never opens or closes on a few frames of another shot."""
    normal = [s for s in segments if not s.get("replay")]
    for i, seg in enumerate(normal):
        start, end = float(seg["start_time"]), float(seg["end_time"])
        span = end - start
        low = float(normal[i - 1]["end_time"]) if i else 0.0
        high = float(normal[i + 1]["start_time"]) if i + 1 < len(normal) else source_duration
        center = float(seg.get("event_center", (start + end) / 2.0))
        head = [c for c in cuts if start < c < start + sliver]
        tail = [c for c in cuts if end - sliver < c < end]
        if head and not tail:
            new_start = max(head) + 0.04
            if new_start + span <= high and new_start <= center:
                seg["start_time"], seg["end_time"] = round(new_start, 3), round(new_start + span, 3)
        elif tail and not head:
            new_end = min(tail) - 0.04
            if new_end - span >= low and new_end >= center:
                seg["start_time"], seg["end_time"] = round(new_end - span, 3), round(new_end, 3)


def _snap_to_beat(segments: list[dict[str, Any]], beat: float, source_duration: float) -> None:
    """Move every cut onto the music's beat grid, keeping the total length.

    Cuts that land on a beat read as edited to the music.  Only output
    durations move (each by less than half a beat); source windows follow.
    """
    if len(segments) < 2 or beat <= 0:
        return
    total = sum(float(seg["output_duration"]) for seg in segments)
    cuts, edge = [], 0.0
    for seg in segments[:-1]:
        edge += float(seg["output_duration"])
        cuts.append(edge)
    snapped, prev = [], 0.0
    for i, cut in enumerate(cuts):
        lo = prev + 0.6
        hi = total - 0.6 * (len(cuts) - i)
        value = min(max(round(cut / beat) * beat, lo), hi)
        snapped.append(value)
        prev = value
    bounds = [0.0, *snapped, total]
    for seg, a, b in zip(segments, bounds, bounds[1:]):
        out = round(b - a, 3)
        speed = float(seg.get("speed", 1.0) or 1.0)
        span = out * speed
        if seg.get("replay"):
            mid = (float(seg["start_time"]) + float(seg["end_time"])) / 2.0
            start = min(max(0.0, mid - span / 2.0), max(0.0, source_duration - span))
        else:
            start = min(float(seg["start_time"]), max(0.0, source_duration - span))
        seg["start_time"] = round(start, 3)
        seg["end_time"] = round(min(source_duration, start + span), 3)
        seg["output_duration"] = out


def build_montage(
    items: list[dict[str, Any]],
    target: float,
    source_duration: float,
    style: dict[str, Any] | None = None,
    max_clips: int | None = None,
    shot_cuts: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Turn ranked candidates into a timed, non-overlapping highlight montage.

    Every returned segment has ``start_time``/``end_time`` (source seconds),
    ``output_duration`` (seconds in the final MP4) and an optional ``speed``.
    The sum of output durations is exactly ``target`` (minus floating noise).
    """
    style = normalize_edit_style(style)
    cfg = PACE_PRESETS[style["pace"]]
    target = max(0.1, min(float(target), max(0.1, float(source_duration))))
    limit = int(max_clips or cfg["max_clips"])
    desired = max(1, round(target / cfg["segment"]))
    chosen = _select_for_montage(items, min(limit, desired), max(1.6, cfg["segment"] * 0.9))
    if not chosen:
        return []

    # Chronological boundaries give each play its own slice of the source.
    centers = [_center_of(item) for item in chosen]
    limits: list[tuple[float, float]] = []
    for index in range(len(chosen)):
        left = 0.0 if index == 0 else (centers[index - 1] + centers[index]) / 2.0
        right = source_duration if index == len(chosen) - 1 else (centers[index] + centers[index + 1]) / 2.0
        limits.append((max(0.0, left), min(source_duration, right)))

    windows: list[list[float]] = []
    for index, item in enumerate(chosen):
        left, right = limits[index]
        lo = max(left, float(item["start_time"]) - cfg["pre"])
        hi = min(right, float(item["end_time"]) + cfg["post"])
        if hi - lo < cfg["min"]:
            mid = (lo + hi) / 2.0
            lo = max(left, mid - cfg["min"] / 2.0)
            hi = min(right, lo + cfg["min"])
            lo = max(left, hi - cfg["min"])
        windows.append([lo, hi])
    _expand_windows(windows, limits, target)

    # Optional slow-motion replay of the strongest play.  It deliberately
    # reuses a short source window, exactly like a broadcast replay.
    replay: dict[str, Any] | None = None
    replay_output = 0.0
    if style["slow_motion"] and target >= 9.0:
        anchor = max(chosen, key=_score_of)
        anchor_window = windows[chosen.index(anchor)]
        span = min(REPLAY_MAX_SOURCE, anchor_window[1] - anchor_window[0])
        if span >= 1.2:
            center = min(max(_center_of(anchor), anchor_window[0] + span / 2.0), anchor_window[1] - span / 2.0)
            replay = {
                "start_time": round(center - span / 2.0, 3),
                "end_time": round(center + span / 2.0, 3),
                "speed": REPLAY_SPEED,
                "output_duration": round(span / REPLAY_SPEED, 3),
                "replay": True,
                "anchor_event_id": anchor.get("event_id"),
                "label": "慢放回看 · " + (anchor.get("display_name") or EVENT_TYPE_ZH.get(anchor.get("event_type", ""), "精彩动作")),
                "event_type": anchor.get("event_type", "tennis_action_candidate"),
                "raw_model_score": _score_of(anchor),
            }
            replay_output = float(replay["output_duration"])
            # Replays are precious but should never crowd out the actual plays.
            if target - replay_output < len(chosen) * 1.2:
                replay = None
                replay_output = 0.0

    content_target = max(0.0, target - replay_output)
    caps = [max(0.1, hi - lo) for lo, hi in windows]
    scores = [_score_of(item) for item in chosen]
    low, high = min(scores), max(scores)
    weights = [0.8 + 0.6 * ((score - low) / (high - low) if high > low else 0.5) for score in scores]
    allocs = _weighted_fill(caps, weights, content_target)
    if allocs:
        allocs[-1] = max(0.0, allocs[-1] + (content_target - sum(allocs)))

    segments: list[dict[str, Any]] = []
    for index, item in enumerate(chosen):
        lo, hi = windows[index]
        out = max(0.2, min(float(allocs[index]), hi - lo))
        center = min(max(_center_of(item), lo + out / 2.0), hi - out / 2.0)
        start = round(center - out / 2.0, 3)
        end = round(start + out, 3)
        if end > source_duration:
            end = round(source_duration, 3)
            start = round(max(lo, end - out), 3)
        segments.append({
            "event_id": item.get("event_id"),
            "start_time": start,
            "end_time": end,
            "output_duration": round(end - start, 3),
            "event_center": round(_center_of(item), 3),
            "speed": 1.0,
            "replay": False,
            "event_type": item.get("event_type", "tennis_action_candidate"),
            "label": item.get("display_name") or EVENT_TYPE_ZH.get(item.get("event_type", ""), "网球动作"),
            "display_name": item.get("display_name") or EVENT_TYPE_ZH.get(item.get("event_type", ""), "网球动作"),
            "action_label_zh": item.get("action_label_zh"),
            "outcome": item.get("outcome"),
            "raw_model_score": _score_of(item),
            "verification_status": item.get("verification_status", "MODEL_PREDICTED"),
        })

    segments = _merge_jump_cuts(segments, shot_cuts, source_duration)
    if replay:
        # Insert the replay straight after the play it revisits.
        anchor_id = replay.get("anchor_event_id")
        insert_at = next((i + 1 for i, seg in enumerate(segments)
                          if anchor_id in (seg.get("event_id"), *seg.get("merged_event_ids", []))), len(segments))
        segments.insert(insert_at, replay)

    # Built-in music has a known tempo: cut on its beat.
    from .music import PRESETS
    if style.get("music") in PRESETS:
        _snap_to_beat(segments, 60.0 / PRESETS[style["music"]]["bpm"], source_duration)

    # Make the output exactly match the requested duration, even after rounding.
    diff = target - sum(float(seg["output_duration"]) for seg in segments)
    if abs(diff) > 1e-6 and segments:
        candidates = [i for i, seg in enumerate(segments) if not seg.get("replay")] or list(range(len(segments)))
        index = candidates[-1]
        seg = segments[index]
        new_out = max(0.2, float(seg["output_duration"]) + diff)
        seg["output_duration"] = round(new_out, 3)
        if not seg.get("replay"):
            seg["end_time"] = round(min(source_duration, float(seg["start_time"]) + new_out), 3)
    if shot_cuts:
        _avoid_slivers(segments, shot_cuts, source_duration)
    return segments


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _fingerprint(video: Path) -> str:
    """Cheap content identity check so replaced source videos invalidate caches."""
    stat = video.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _video_info(video: Path) -> dict[str, Any]:
    try:
        raw = subprocess.check_output([str(FFPROBE), "-v", "error", "-show_entries", "format=duration:stream=codec_type,r_frame_rate,width,height", "-of", "json", str(video)], text=True, timeout=30)
        data = json.loads(raw)
        return {"duration": float(data["format"]["duration"]), "streams": data.get("streams", []), "probe": data}
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"cannot inspect {video.name}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration = frames / fps if fps > 0 else 0.0
        if duration <= 0:
            raise RuntimeError(f"cannot determine duration for {video.name}")
        return {"duration": duration, "streams": [{"codec_type": "video", "width": width, "height": height}], "probe": {"fallback": "opencv"}}


class X3D:
    """Official PyTorchVideo Hub X3D-XS, cached strictly inside this project."""
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = {int(v): k.strip('"') for k, v in _json(LABELS_PATH, {}).items()}
        if not self.labels:
            raise RuntimeError("missing official Kinetics label cache")
        os.environ["TORCH_HOME"] = str(TORCH_HOME)
        before = torch.cuda.memory_allocated() if self.device.type == "cuda" else 0
        started = time.perf_counter()
        # skip_validation avoids GitHub API credentials; Hub source and checkpoint
        # are already downloaded from their official URLs into project cache.
        # The explicit ":main" ref matters: without it torch.hub asks GitHub for
        # the default branch on every load, and a GitHub outage breaks analysis.
        self.model = torch.hub.load("facebookresearch/pytorchvideo:main", "x3d_xs", pretrained=True, trust_repo=True, skip_validation=True).eval().to(self.device)
        self.load_seconds = round(time.perf_counter() - started, 3)
        self.memory_bytes = int(torch.cuda.memory_allocated() - before) if self.device.type == "cuda" else 0

    def _input(self, frames: list[np.ndarray]) -> torch.Tensor:
        # Hub documentation for X3D-XS: 4 frames x sample rate 12, crop 182.
        tensor = torch.from_numpy(np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in frames])).permute(3, 0, 1, 2).float() / 255.0
        tensor = (tensor - torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]) / torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
        # Resize short side to 182 then center crop: no torchvision-video API
        # dependency, and exactly the documented X3D spatial preprocessing.
        _, _, h, w = tensor.shape
        scale = 182 / min(h, w)
        tensor = torch.nn.functional.interpolate(tensor.permute(1, 0, 2, 3), size=(round(h * scale), round(w * scale)), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        _, _, h, w = tensor.shape
        y, x = (h - 182) // 2, (w - 182) // 2
        return tensor[:, :, y:y + 182, x:x + 182].unsqueeze(0).to(self.device)

    def classify_detailed(self, frames: list[np.ndarray]) -> dict[str, Any]:
        tensor = self._input(frames)
        with torch.inference_mode():
            logits = self.model(tensor)[0]
            scores = torch.softmax(logits, dim=0)
        values, indices = scores.topk(5)
        top5 = [{"label": self.labels.get(int(i), f"class_{int(i)}"), "class_id": int(i),
                 "logit": round(float(logits[int(i)]), 6), "softmax_score": round(float(v), 6)}
                for v, i in zip(values.detach().cpu(), indices.detach().cpu())]
        return {"top5": top5, "input_shape": list(tensor.shape), "score_type": "softmax(logits)"}

    def classify(self, frames: list[np.ndarray]) -> list[dict[str, Any]]:
        """V1 compatibility: retain its legacy field while V2 uses detailed audit."""
        return [{"label": x["label"], "class_id": x["class_id"], "raw_model_score": x["softmax_score"]}
                for x in self.classify_detailed(frames)["top5"]]


# One model per process: the web server runs jobs in threads, and loading X3D
# per job wasted seconds and GPU memory.  ANALYSIS_LOCK serialises scans so two
# jobs on an uncached video do not both scan it and race on latest_<id>.json.
ANALYSIS_LOCK = threading.RLock()
_X3D_LOCK = threading.Lock()
_X3D_INSTANCE: X3D | None = None


def get_x3d() -> X3D:
    global _X3D_INSTANCE
    with _X3D_LOCK:
        if _X3D_INSTANCE is None:
            _X3D_INSTANCE = X3D()
        return _X3D_INSTANCE


def _frames(video: Path, start: float, duration: float = 1.6) -> tuple[list[np.ndarray], list[float]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode {video.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Some WebM containers report a timebase (for example 30000) as FPS.
    if fps > 240:
        fps = 30.0
    frames, times = [], []
    for offset in (0.0, duration / 3, duration * 2 / 3, duration):
        t = start + offset
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
        times.append(round(t, 3))
    cap.release()
    if len(frames) != 4:
        raise RuntimeError("insufficient decodable frames for X3D window")
    return frames, times


def _motion_events(video_id: str, duration: float) -> list[dict[str, Any]]:
    """Offline fallback: sustained court motion gives highlighter candidates, never point calls."""
    cap = cv2.VideoCapture(str(VIDEO_IDS[video_id]))
    if not cap.isOpened():
        return []
    samples: list[tuple[float, float]] = []
    previous = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Some WebM containers report a timebase (for example 30000) as FPS.
    if fps > 240:
        fps = 30.0
    skip = max(1, round(fps * 0.4))
    last_t = -0.4
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        if t - last_t < 0.4:
            continue
        last_t = t
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if previous is not None:
            samples.append((t, float(cv2.absdiff(gray, previous).mean())))
        previous = gray
        for _ in range(skip - 1):
            if not cap.grab():
                break
    cap.release()
    if not samples:
        return []
    values = np.array([value for _, value in samples])
    threshold = max(float(np.percentile(values, 70)), float(np.median(values) * 1.2), 2.0)
    events = []
    for i, (t, score) in enumerate(samples):
        left = samples[i - 1][1] if i else -1.0
        right = samples[i + 1][1] if i + 1 < len(samples) else -1.0
        if score < threshold or score < left or score < right:
            continue
        events.append({"event_id": f"motion-{video_id}-{round(t * 10):05d}", "source_video_id": video_id,
            "video_path": str(VIDEO_IDS[video_id].relative_to(ROOT)), "event_type": "tennis_action_candidate",
            "action_label": "court_motion", "action_label_zh": "精彩回合候选", "display_name": "精彩回合候选",
            "start_time": round(max(0.0, t - 1.0), 3), "end_time": round(min(duration, t + 1.4), 3),
            "raw_model_score": round(min(1.0, score / max(float(values.max()), 1.0)), 6), "motion_score": round(score, 4),
            "score_kind": "court-motion heuristic; not a tennis action probability", "source_model": "OpenCV court-motion fallback",
            "evidence_source": "LOCAL_MOTION", "verification_status": "MODEL_PREDICTED",
            "semantic_limit": "Motion candidates do not verify a serve, point, score, player, winner, or rally outcome."})
    return _merge(events)


def _hype_segments(video_id: str) -> list[dict[str, Any]]:
    """Return the primary HypeCut candidate segments for a video.

    ``default.json`` is the plan produced by the normal baseline flow.  Older
    checkouts only had profile-specific files (e.g. ``sports-broadcast.json``),
    so fall back to the richest available profile instead of silently returning
    an empty HypeCut-only timeline.
    """
    if video_id not in HYPECUT:  # uploads have no HypeCut baseline
        return []
    primary = _json(HYPECUT[video_id], {})
    segments = primary.get("segments") if isinstance(primary, dict) else None
    if isinstance(segments, list) and segments:
        return segments
    best: list[dict[str, Any]] = []
    directory = HYPECUT_FALLBACK_DIRS[video_id]
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            data = _json(path, {})
            candidate = data.get("segments") if isinstance(data, dict) else None
            if isinstance(candidate, list) and len(candidate) > len(best):
                best = candidate
    return best


def _windows(video_id: str, duration: float) -> list[tuple[float, str, dict[str, Any] | None]]:
    # 4 s global cadence, augmented at 2 s cadence near HypeCut candidates.
    # Four-second sampling keeps the local scan affordable while giving the
    # editor enough temporal candidates to fill a 20-second request.
    merged: dict[int, tuple[float, str, dict[str, Any] | None]] = {}
    # Without HypeCut hints (uploads) scan twice as densely so no play is missed.
    step = 4.0 if video_id in HYPECUT else 2.0
    global_times = list(np.arange(0.0, max(0.1, duration - 1.7), step))
    if duration > 1.7:
        global_times.append(max(0.0, duration - 1.7))
    for t in global_times:
        merged[round(float(t) * 10)] = (float(t), "full_video_low_frequency", None)
    for segment in _hype_segments(video_id):
        for t in np.arange(float(segment["start"]), float(segment["end"]), 2.0):
            key = round(float(t) * 10)
            merged[key] = (float(t), "hypecut_dense", segment)
    return [merged[k] for k in sorted(merged)]


def _event_from_window(video_id: str, start: float, times: list[float], top: list[dict[str, Any]], source: str, segment: dict[str, Any] | None) -> dict[str, Any] | None:
    """Turn Kinetics tennis labels into highlighter candidates, never point calls."""
    tennis = [x for x in top if x["label"] in TENNIS_LABELS]
    if not tennis:
        return None
    primary = max(tennis, key=lambda x: float(x["raw_model_score"]))
    event_type, label_zh = TENNIS_LABELS[primary["label"]]
    signals = {x["label"]: round(float(x["raw_model_score"]), 6) for x in tennis}
    return {
        "event_id": f"lw-{video_id}-{round(start * 10):05d}",
        "source_video_id": video_id,
        "video_path": str(VIDEO_IDS[video_id].relative_to(ROOT)),
        "event_type": event_type,
        "action_label": primary["label"],
        "action_label_zh": label_zh,
        "display_name": EVENT_TYPE_ZH.get(event_type, label_zh),
        "start_time": round(times[0], 3),
        "end_time": round(times[-1], 3),
        "raw_model_score": primary["raw_model_score"],
        "tennis_signals": signals,
        "score_kind": "Kinetics-400 softmax; not a calibrated tennis-event probability",
        "source_model": "PyTorchVideo X3D-XS (Kinetics-400)",
        "model_version": "pytorchvideo hub main / X3D_XS.pyth",
        "visual_action_evidence": {"top5": top, "matched_label": primary["label"], "frame_timestamps": times},
        "hypecut_signal": {"sampling": source, "segment_score": segment.get("score") if segment else None, "reasons": segment.get("reasons") if segment else {}},
        "evidence_source": "LOCAL_X3D",
        "verification_status": "MODEL_PREDICTED",
        "semantic_limit": "Generic Kinetics action evidence only; this does not verify a point, score, player, winner, or rally outcome.",
        "time_localization_method": "1.6s X3D sample window",
    }


def _merge(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda x: (x["start_time"], -x["raw_model_score"])):
        if result and event["event_type"] == result[-1]["event_type"] and event["start_time"] - result[-1]["end_time"] <= 2.0:
            if event["raw_model_score"] > result[-1]["raw_model_score"]:
                result[-1] = event
            continue
        result.append(event)
    return result


def _fresh_cache(video_id: str) -> dict[str, Any] | None:
    # A cache is reusable only when it was produced by the current analysis
    # pipeline and the source file has not changed.  Legacy caches without
    # a version are intentionally re-analysed so users never see stale sport labels.
    cached = _json(OUT / f"latest_{video_id}.json", {})
    if cached.get("analysis_version") == ANALYSIS_VERSION and cached.get("source_fingerprint") == _fingerprint(VIDEO_IDS[video_id]):
        return cached
    return None


def analyze(video_id: str, force: bool = False, progress: Any = None) -> dict[str, Any]:
    """``progress(done, total)`` is called while scanning, if given."""
    if video_id not in VIDEO_IDS:
        raise ValueError("unknown video id")
    if not force and (cached := _fresh_cache(video_id)):
        return cached
    with ANALYSIS_LOCK:
        # Another job may have finished the scan while this one waited.
        if not force and (cached := _fresh_cache(video_id)):
            return cached
        return _analyze_locked(video_id, progress)


def _analyze_locked(video_id: str, progress: Any = None) -> dict[str, Any]:
    latest = OUT / f"latest_{video_id}.json"
    video = VIDEO_IDS[video_id]
    fingerprint = _fingerprint(video)
    OUT.mkdir(parents=True, exist_ok=True)
    info = _video_info(video)
    started = time.perf_counter()
    runner = None
    model_error = None
    try:
        runner = get_x3d()
    except Exception as exc:
        model_error = type(exc).__name__
    raw, failures = [], []
    windows = _windows(video_id, info["duration"])
    x3d_share = 100
    for number, (start, sampling, segment) in enumerate(windows):
        if progress and number % 4 == 0:
            progress(round(100 * number / max(1, len(windows))), 100)
        try:
            if runner is not None:
                frames, times = _frames(video, start)
                top = runner.classify(frames)
                event = _event_from_window(video_id, start, times, top, sampling, segment)
                if event:
                    raw.append(event)
        except Exception as exc:
            failures.append({"start_time": start, "error": type(exc).__name__})
    x3d_seconds = time.perf_counter() - started
    events = _merge(raw) if runner is not None else _motion_events(video_id, info["duration"])
    try:
        from integrations import highlight
        highlight.score(events, highlight.loudness(video, FFMPEG))
    except Exception:  # ranking is an extra; the old scores stand
        pass
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": "1.1", "analysis_version": ANALYSIS_VERSION, "run_id": f"lw-{video_id}-{_now()}", "source_video_id": video_id,
        "source_video": str(video.relative_to(ROOT)), "source_fingerprint": fingerprint, "events": events,
        "event_summary": {kind: sum(1 for event in events if event["event_type"] == kind) for kind in sorted({event["event_type"] for event in events})},
        "shooting": None,
        "roster": {"players": [], "teams": False},
        "detector": {"available": runner is not None, "reason": "X3D unavailable; using court-motion candidates" if model_error else "Basketball rim detector is disabled for tennis", "model_error": model_error, "x3d_seconds": round(x3d_seconds, 3)},
        "hypecut_only_baseline": {"candidate_count": len(_hype_segments(video_id)), "segments": _hype_segments(video_id)},
        "performance": {"device": str(runner.device) if runner is not None else "opencv", "model_load_seconds": runner.load_seconds if runner is not None else 0.0, "gpu_memory_model_bytes": runner.memory_bytes if runner is not None else 0,
                        "inference_total_seconds": round(elapsed, 3), "seconds_per_video_minute": round(elapsed / (info["duration"] / 60), 3),
                        "windows": len(_windows(video_id, info["duration"])), "raw_candidates": len(raw), "merged_candidates": len(events), "failures": failures},
        "limitations": "No human ground truth was used: Precision/Recall/F1 are UNVERIFIED.",
    }
    stamped = OUT / f"{payload['run_id']}.json"
    stamped.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Readers (events(), the web API) never see a half-written cache.
    partial = latest.with_suffix(".json.part")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(partial, latest)
    return payload


def events(video_id: str, analyze_if_missing: bool = False, progress: Any = None) -> list[dict[str, Any]]:
    """Analysed events with the user's 复盘 corrections applied."""
    from integrations import corrections
    return corrections.apply(video_id, _raw_events(video_id, analyze_if_missing, progress))


def _raw_events(video_id: str, analyze_if_missing: bool = False, progress: Any = None) -> list[dict[str, Any]]:
    if video_id not in VIDEO_IDS:
        raise ValueError("unknown video id")
    latest = OUT / f"latest_{video_id}.json"
    cached = _json(latest, None)
    if cached is None:
        return analyze(video_id, progress=progress).get("events", []) if analyze_if_missing else []
    fingerprint = cached.get("source_fingerprint")
    stale = cached.get("analysis_version") != ANALYSIS_VERSION or (fingerprint and fingerprint != _fingerprint(VIDEO_IDS[video_id]))
    if stale:
        return analyze(video_id, force=True, progress=progress).get("events", []) if analyze_if_missing else []
    return cached.get("events", [])


def _clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _filter_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _probe_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(subprocess.check_output(
            [str(FFPROBE), "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name",
             "-of", "json", str(path)], text=True, timeout=30))
    except (OSError, subprocess.SubprocessError, ValueError):
        # The project-local imageio FFmpeg runtime ships the encoder binary but
        # not ffprobe. OpenCV still provides the duration and stream shape we
        # need for rendering, so a missing optional probe must not block edits.
        info = _video_info(path)
        return {"format": {"duration": str(info["duration"])}, "streams": info["streams"]}


CANVAS = {"16:9": (1280, 720), "9:16": (720, 1280)}
RENDER_MARGIN = 0.1  # extra source so trims never run short of frames


def _crossfade_seconds(style: dict[str, Any], outs: list[float]) -> float:
    """Overlap between clips; 0 means a straight cut."""
    base = {"fade": 0.32, "flash": 0.28}.get(style["transition"], 0.0)
    if not base or len(outs) < 2:
        return 0.0
    return round(min(base, 0.4 * min(outs)), 3)


def _pad_window(seg: dict[str, Any], head: float, tail: float, margin: float, duration: float) -> dict[str, Any]:
    """Copy of a segment that also covers its crossfade overlaps (output seconds)."""
    speed = float(seg.get("speed", 1.0) or 1.0)
    start = float(seg["start_time"]) - head * speed
    end = float(seg["end_time"]) + (tail + margin) * speed
    if start < 0:
        end, start = end - start, 0.0
    if end > duration:
        start, end = max(0.0, start - (end - duration)), duration
    out = dict(seg)
    out.update(start_time=round(start, 3), end_time=round(end, 3),
               output_duration=round(float(seg["output_duration"]) + head + tail + margin, 3))
    return out


def _join_crossfade(paths: list[Path], lengths: list[float], outs: list[float], d: float,
                    transition: str, target: float, destination: Path) -> None:
    """Join clips with real dissolves (or white flashes) centred on each cut.

    Clip i is trimmed to ``lengths[i]`` = planned length + half an overlap on
    each joined side, so every transition straddles the planned cut point and
    the reel is still exactly ``target`` long.
    """
    kind = "fadewhite" if transition == "flash" else "fade"
    graph = []
    for i, length in enumerate(lengths):
        graph.append(f"[{i}:v]trim=duration={length:.3f},setpts=PTS-STARTPTS,fps=30,format=yuv420p,settb=AVTB[v{i}]")
        graph.append(f"[{i}:a]atrim=duration={length:.3f},asetpts=PTS-STARTPTS,"
                     f"aformat=sample_rates=48000:channel_layouts=stereo[a{i}]")
    video, audio, cut = "v0", "a0", 0.0
    for i in range(1, len(paths)):
        cut += outs[i - 1]
        graph.append(f"[{video}][v{i}]xfade=transition={kind}:duration={d:.3f}:offset={max(0.0, cut - d / 2):.3f}[vx{i}]")
        graph.append(f"[{audio}][a{i}]acrossfade=d={d:.3f}:c1=tri:c2=tri[ax{i}]")
        video, audio = f"vx{i}", f"ax{i}"
    cmd = [str(FFMPEG), "-y", "-v", "error"]
    for path in paths:
        cmd += ["-i", str(path)]
    cmd += ["-filter_complex", ";".join(graph), "-map", f"[{video}]", "-map", f"[{audio}]", "-t", f"{target:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(destination)]
    subprocess.run(cmd, check=True, timeout=900)


def _subject_track(video: Path, start: float, end: float) -> tuple[float, float]:
    """Horizontal centre of the action (0–1 of frame width) near a clip's start and end.

    Motion between two frames 0.12 s apart marks where players move; blending
    with the frame centre keeps a camera pan from yanking the vertical crop.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0.5, 0.5
    points = []
    try:
        for k in range(5):
            t = start + (end - start) * (k + 0.5) / 5
            pair = []
            for offset in (0.0, 0.12):
                cap.set(cv2.CAP_PROP_POS_MSEC, (t + offset) * 1000)
                ok, frame = cap.read()
                if not ok:
                    break
                pair.append(cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32))
            if len(pair) == 2:
                energy = np.abs(pair[1] - pair[0])[12:78].sum(axis=0)  # skip score bugs at the edges
                if energy.sum() > 1e-3:
                    points.append(float((energy * np.arange(160)).sum() / energy.sum() / 160))
    finally:
        cap.release()
    if not points:
        return 0.5, 0.5
    early = float(np.mean(points[: max(1, len(points) // 2)]))
    late = float(np.mean(points[len(points) // 2:]))
    x0, x1 = (0.6 * early + 0.2, 0.6 * late + 0.2)  # 60 % motion, 40 % centre prior
    if abs(x1 - x0) > 0.25:  # cap the pan speed
        mid = (x0 + x1) / 2
        x0, x1 = mid - 0.125 * np.sign(x1 - x0), mid + 0.125 * np.sign(x1 - x0)
    return round(min(max(x0, 0.0), 1.0), 3), round(min(max(x1, 0.0), 1.0), 3)


LOOK_FILTERS = {
    "vivid": ["eq=contrast=1.055:saturation=1.13:brightness=0.008"],
    "cinematic": ["eq=contrast=1.12:saturation=0.86:brightness=-0.01",
                  "colorbalance=rs=-0.06:bs=0.08:rh=0.08:bh=-0.07", "vignette=PI/4.5"],
    "retro": ["curves=preset=vintage", "eq=saturation=0.85", "noise=alls=10:allf=t", "vignette=PI/4"],
    "bw": ["hue=s=0", "eq=contrast=1.25:brightness=-0.02", "vignette=PI/5"],
}
ZOOM_AMOUNT = 0.07          # gentle push-in across a clip
REPLAY_ZOOM_AMOUNT = 0.14   # stronger on the slow-motion replay


def _segment_filter(
    seg: dict[str, Any], index: int, total: int, style: dict[str, Any],
    temp_dir: Path,
) -> str:
    out_dur = max(0.2, float(seg["output_duration"]))
    src_dur = max(0.1, float(seg["end_time"]) - float(seg["start_time"]))
    # ``setpts`` multiplies timestamps: >1 slows the clip down.
    pts = max(0.05, out_dur / src_dur)
    width, height = CANVAS.get(style.get("aspect", "16:9"), CANVAS["16:9"])
    if height > width:
        # Vertical: fill the frame and pan the crop with the action.
        x0, x1 = seg.get("track", (0.5, 0.5))
        centre = f"({x0:.3f}+({x1 - x0:.3f})*min(1,t/{src_dur:.3f}))"
        parts = [f"scale={width}:{height}:force_original_aspect_ratio=increase",
                 f"crop={width}:{height}:x='clip({centre}*iw-{width}/2,0,iw-{width})':y='(ih-{height})/2'",
                 "setsar=1"]
    else:
        parts = [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
            "setsar=1",
        ]
    if abs(pts - 1.0) > 0.005:
        parts.append(f"setpts={pts:.6f}*PTS")
    parts.append("fps=30")
    if style.get("zoom") or seg.get("replay"):
        # Ken Burns push-in.  Upscaling first keeps zoompan's integer crop from
        # wobbling; d=1 keeps one output frame per input frame.
        amount = REPLAY_ZOOM_AMOUNT if seg.get("replay") else ZOOM_AMOUNT
        frames = max(1, round(out_dur * 30))
        parts.append(f"scale={2 * width}:{2 * height}")
        parts.append(f"zoompan=z='1+{amount}*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps=30")
    look = style.get("look", "vivid")
    parts.extend(LOOK_FILTERS.get(look, LOOK_FILTERS["vivid"]))

    # Clips are joined by real crossfades (_join_crossfade); only the reel
    # itself opens from and closes to black.
    if index == 0:
        parts.append("fade=t=in:st=0:d=0.35")
    if index == total - 1:
        parts.append(f"fade=t=out:st={max(0.0, out_dur - 0.38):.3f}:d=0.38")

    letterbox = look == "cinematic" and width > height
    if letterbox:
        parts.append("drawbox=x=0:y=0:w=iw:h=ih*0.1:color=black:t=fill")
        parts.append("drawbox=x=0:y=ih*0.9:w=iw:h=ih*0.1:color=black:t=fill")
    lift = 64 if letterbox else (190 if height > width else 0)  # clear of short-video app UI

    font = FONT_CJK if FONT_CJK.is_file() else FONT_LATIN
    if font.is_file():
        title = style.get("title")
        if index == 0 and (title or (title is None and style["captions"])):
            text = title or "MoTrack 高光"
            title_file = temp_dir / "title.txt"
            title_file.write_text(text, encoding="utf-8")
            size = max(24, min(46, int(width * 0.86 / max(4, len(text)))))
            parts.append(
                "drawtext=textfile='" + _filter_path(title_file) + "':expansion=none"
                + ":fontfile='" + _filter_path(font) + "'"
                + f":fontcolor=white@0.94:fontsize={size}:x=(w-tw)/2:y=h*0.17"
                + ":borderw=2:bordercolor=black@0.45"
                + ":enable='lt(t,2.4)'"
            )
        if style["captions"]:
            label = str(seg.get("label") or seg.get("display_name") or "精彩片段")
            caption = temp_dir / f"cap_{index:02d}.txt"
            caption.write_text(f"{label}   {_clock(float(seg['start_time']))}", encoding="utf-8")
            parts.append(f"drawbox=x=34:y=ih-{126 + lift}:w=7:h=70:color=0xFF7A18@0.95:t=fill")
            parts.append(
                "drawtext=textfile='" + _filter_path(caption) + "':expansion=none"
                + ":fontfile='" + _filter_path(font) + "'"
                + f":fontcolor=white:fontsize=27:x=58:y=h-{122 + lift}"
                + ":box=1:boxcolor=black@0.42:boxborderw=13"
                + ":shadowcolor=black@0.55:shadowx=1:shadowy=2"
            )
    if FONT_LATIN.is_file():
        parts.append(
            "drawtext=text='MOTRACK':fontfile='" + _filter_path(FONT_LATIN) + "'"
            + f":fontcolor=white@0.40:fontsize=16:x=w-tw-30:y={26 + (8 if letterbox else 0)}"
            + ":shadowcolor=black@0.5:shadowx=1:shadowy=1"
        )
    return ",".join(parts)


def _render_segment(
    video: Path, seg: dict[str, Any], index: int, total: int, style: dict[str, Any],
    temp_dir: Path, destination: Path, has_audio: bool,
) -> None:
    src_dur = max(0.1, float(seg["end_time"]) - float(seg["start_time"]))
    out_dur = max(0.2, float(seg["output_duration"]))
    speed = max(0.2, min(2.0, float(seg.get("speed", 1.0) or 1.0)))
    vf = _segment_filter(seg, index, total, style, temp_dir)
    width, height = CANVAS.get(style.get("aspect", "16:9"), CANVAS["16:9"])

    def command(with_look: bool) -> list[str]:
        cmd = [str(FFMPEG), "-y", "-v", "error",
               "-ss", f"{float(seg['start_time']):.3f}", "-t", f"{src_dur:.3f}", "-i", str(video)]
        if not has_audio:
            cmd += ["-f", "lavfi", "-t", f"{out_dur:.3f}", "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000"]
        filters = vf if with_look else (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
            "setpts=" + f"{max(0.05, out_dur / src_dur):.6f}" + "*PTS,fps=30"
        )
        cmd += ["-map", "0:v:0"]
        audio_filters = []
        if has_audio:
            cmd += ["-map", "0:a:0"]
            if abs(speed - 1.0) > 0.005:
                audio_filters.append(f"atempo={speed:.4f}")
            if style["transition"] == "cut":  # crossfades already smooth the joins
                audio_filters.append("afade=t=in:st=0:d=0.03")
                audio_filters.append(f"afade=t=out:st={max(0.0, out_dur - 0.05):.3f}:d=0.05")
        else:
            cmd += ["-map", "1:a:0"]
        cmd += ["-vf", filters]
        if audio_filters:
            cmd += ["-af", ",".join(audio_filters)]
        cmd += ["-t", f"{out_dur:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
                "-movflags", "+faststart", str(destination)]
        return cmd

    try:
        subprocess.run(command(True), check=True, timeout=600)
    except subprocess.CalledProcessError:
        # A missing font or an exotic source must never break the whole render.
        subprocess.run(command(False), check=True, timeout=600)


def render_plan(video_id: str, plan: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Render a real multi-clip montage with trims, transitions and captions."""
    clips = plan.get("clips", [])
    if not clips:
        raise ValueError("empty highlight plan")
    info = _video_info(VIDEO_IDS[video_id])
    duration = float(info["duration"])
    has_audio = any(stream.get("codec_type") == "audio" for stream in info.get("streams", []))
    for clip in clips:
        # Clamp model-generated context to the real source length.  A candidate
        # at the very end of the video otherwise gets ``end + 2s`` and fails
        # validation, which previously broke HypeCut-only on short clips.
        start_time = max(0.0, min(float(clip["start_time"]), duration - 0.1))
        end_time = min(duration, max(start_time + 0.1, float(clip["end_time"])))
        clip["start_time"], clip["end_time"] = round(start_time, 3), round(end_time, 3)

    style = normalize_edit_style(plan.get("edit_style"))
    requested = float(plan.get("requested_duration", plan.get("target_duration", max(x["end_time"] for x in clips) - min(x["start_time"] for x in clips))))
    meta = duration_meta(requested, duration)
    # Preserve the caller's clamp explanation (e.g. 30 s requested on a 20.5 s
    # source) instead of re-deriving it from the already-capped target.
    if "effective_duration" in plan and abs(float(plan["effective_duration"]) - meta["effective_duration"]) <= 0.05:
        meta["duration_clamped"] = bool(plan.get("duration_clamped", meta["duration_clamped"]))
        meta["duration_note"] = plan.get("duration_note") or meta["duration_note"]
        meta["requested_duration"] = float(plan.get("requested_duration", meta["requested_duration"]))
    target = meta["effective_duration"]

    precomputed = all(clip.get("output_duration") for clip in clips) and abs(
        sum(float(clip["output_duration"]) for clip in clips) - target
    ) <= 0.12
    if precomputed:
        segments = [dict(clip) for clip in clips]
        # Clamp pre-computed durations so they always fit the physical source.
        for seg in segments:
            speed = max(0.2, float(seg.get("speed", 1.0) or 1.0))
            available = (float(seg["end_time"]) - float(seg["start_time"])) / speed
            seg["output_duration"] = round(min(float(seg["output_duration"]), available), 3)
    else:
        segments = build_montage(clips, target, duration, style, max_clips=len(clips))

    if not segments:
        raise ValueError("no renderable highlight segments")
    # Final duration correction: the reel must not drift simply because a
    # plan was assembled by a different caller.
    total = sum(float(seg["output_duration"]) for seg in segments)
    diff = target - total
    if abs(diff) > 1e-6:
        normal = [i for i, seg in enumerate(segments) if not seg.get("replay")] or list(range(len(segments)))
        seg = segments[normal[-1]]
        seg["output_duration"] = round(max(0.2, float(seg["output_duration"]) + diff), 3)
        if not seg.get("replay"):
            seg["end_time"] = round(min(duration, float(seg["start_time"]) + float(seg["output_duration"])), 3)

    destination.parent.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="motrack_render_", dir=str(destination.parent)) as raw_tmp:
        temp_dir = Path(raw_tmp)
        paths: list[Path] = []
        outs = [float(seg["output_duration"]) for seg in segments]
        d = _crossfade_seconds(style, outs)
        lengths: list[float] = []
        for index, seg in enumerate(segments):
            last = index == len(segments) - 1
            head = d / 2 if index else 0.0
            tail = 0.0 if last else d / 2
            padded = _pad_window(seg, head, tail, 0.0 if last else RENDER_MARGIN, duration) if d else dict(seg)
            if style.get("aspect") == "9:16":
                padded["track"] = seg["track"] = _subject_track(VIDEO_IDS[video_id], float(padded["start_time"]), float(padded["end_time"]))
            seg_path = temp_dir / f"seg_{index:02d}.mp4"
            _render_segment(VIDEO_IDS[video_id], padded, index, len(segments), style, temp_dir, seg_path, has_audio)
            paths.append(seg_path)
            lengths.append(outs[index] + head + tail)
        listing = temp_dir / "concat.txt"
        listing.write_text("\n".join(f"file '{path.name}'" for path in paths) + "\n", encoding="utf-8")
        exact = f"{target:.3f}"
        joined = False
        if d:
            try:
                _join_crossfade(paths, lengths, outs, d, style["transition"], target, destination)
                joined = True
            except subprocess.SubprocessError:
                joined = False  # fall back to straight cuts below
        if not joined:
            try:
                subprocess.run([str(FFMPEG), "-y", "-v", "error", "-f", "concat", "-safe", "0",
                                "-i", str(listing), "-t", exact, "-c", "copy", "-movflags", "+faststart", str(destination)],
                               cwd=str(temp_dir), check=True, timeout=600)
            except subprocess.CalledProcessError:
                subprocess.run([str(FFMPEG), "-y", "-v", "error", "-f", "concat", "-safe", "0",
                                "-i", str(listing), "-t", exact, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
                                "-movflags", "+faststart", str(destination)],
                               cwd=str(temp_dir), check=True, timeout=600)

        probe = _probe_file(destination)
        actual = float(probe["format"]["duration"])
        if abs(actual - target) > 0.012:
            # Encoders round each segment to a whole frame, so a six-clip reel
            # can drift by a few hundredths.  One corrective pass keeps the
            # promise: the file is exactly the requested (or source-capped)
            # length.
            corrected = temp_dir / "final.mp4"
            delta = target - actual
            cmd = [str(FFMPEG), "-y", "-v", "error", "-i", str(destination)]
            if delta > 0:
                cmd += ["-vf", f"tpad=stop_mode=clone:stop_duration={delta + 0.25:.3f}", "-af", "apad"]
            cmd += ["-t", f"{target:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
                    "-movflags", "+faststart", str(corrected)]
            subprocess.run(cmd, check=True, timeout=600)
            os.replace(corrected, destination)
            probe = _probe_file(destination)

    music_info = {"music": style.get("music", "none"), "music_applied": False}
    if style.get("music", "none") != "none":
        from .music import resolve, mix_command
        track, _ = resolve(style["music"])
        if track:
            mixed = destination.with_name(destination.stem + f".music.{uuid.uuid4().hex[:6]}.mp4")
            try:
                subprocess.run(mix_command(destination, track, target, has_audio, mixed), check=True, timeout=600)
                os.replace(mixed, destination)
                music_info["music_applied"] = True
                probe = _probe_file(destination)
            except subprocess.SubprocessError:
                mixed.unlink(missing_ok=True)  # keep the reel with its original sound

    output_start = 0.0
    timeline = []
    for index, seg in enumerate(segments):
        out_dur = float(seg["output_duration"])
        timeline.append({
            "index": index + 1,
            "label": seg.get("label") or seg.get("display_name") or EVENT_TYPE_ZH.get(seg.get("event_type", ""), "精彩片段"),
            "event_type": seg.get("event_type"),
            "source_start": round(float(seg["start_time"]), 3),
            "source_end": round(float(seg["end_time"]), 3),
            "output_start": round(output_start, 3),
            "output_end": round(output_start + out_dur, 3),
            "output_duration": round(out_dur, 3),
            "speed": float(seg.get("speed", 1.0)),
            "replay": bool(seg.get("replay")),
        })
        output_start += out_dur

    normal = [seg for seg in segments if not seg.get("replay")] or segments
    return {
        "path": str(destination.relative_to(ROOT)),
        "start": round(min(float(x["start_time"]) for x in normal), 3),
        "end": round(max(float(x["end_time"]) for x in normal), 3),
        "requested_duration": meta["requested_duration"],
        "effective_duration": meta["effective_duration"],
        "source_duration": meta["source_duration"],
        "duration_clamped": meta["duration_clamped"],
        "duration_note": meta["duration_note"],
        "duration": float(probe["format"]["duration"]),
        "probe": probe,
        "render_seconds": round(time.perf_counter() - began, 3),
        "render_mode": "multi_clip_montage",
        "edit_style": style,
        "transition": style["transition"],
        "captions": style["captions"],
        "slow_motion": style["slow_motion"],
        **music_info,
        "segments": timeline,
    }


def _clip_span(clips: list[dict[str, Any]]) -> float:
    if not clips:
        return 0.0
    return max(float(x["end_time"]) for x in clips) - min(float(x["start_time"]) for x in clips)


def fit_clips_to_target(clips: list[dict[str, Any]], target: float, score_key: str = "raw_model_score") -> list[dict[str, Any]]:
    """Drop the weakest clips until the continuous span fits ``target``.

    The renderer only renders one continuous source span.  Without this step a
    plan could list clips that are not actually present in the MP4.  The
    highest-scoring clip is always kept; if a single clip is longer than the
    target its end is clamped.
    """
    target = max(0.1, float(target))
    kept = sorted((dict(x) for x in clips), key=lambda x: float(x["start_time"]))
    if not kept:
        return kept
    while len(kept) > 1 and _clip_span(kept) > target:
        # Remove the lowest-scored clip, preferring one that is not the anchor.
        weakest = min(range(len(kept)), key=lambda i: float(kept[i].get(score_key, kept[i].get("ranking_score", 0.0)) or 0.0))
        kept.pop(weakest)
    kept.sort(key=lambda x: float(x["start_time"]))
    if _clip_span(kept) > target:
        first = kept[0]
        first["end_time"] = min(float(first["end_time"]), float(first["start_time"]) + target)
    return kept


SCORING_WORDS =("发球", "击球", "回合", "网球", "serve", "tennis", "rally")
# Point, set, and winner inference needs ball tracking and scoreboard OCR, neither
# of which is part of this release.
OUTCOME_QUESTION = re.compile(r"(得分|比分|谁赢|赢了吗|几比几|ace|破发)")
HELP_WORDS = ("帮助", "怎么用", "能做什么", "功能", "使用说明")
SEARCH_WORDS = ("找出", "查找", "搜索", "查看", "关键事件", "有哪些", "列出")
RECAP_WORDS = ("解释", "复盘", "分析", "说明", "总结", "表现", "投得怎么样", "哪段时间", "什么时候")
PLAYER_RE = re.compile(r"(?<![第\d])(\d{1,2})\s*号(?:球员|选手|队员)?")
ALL_PLAYERS_WORDS = ("所有人", "全部球员", "全队", "不限球员", "取消只看", "所有球员", "大家的")


def player_from_text(text: str) -> int | None:
    m = PLAYER_RE.search(text or "")
    return int(m.group(1)) if m else None


def roster(video_id: str) -> dict[str, Any]:
    """Players numbered by the last analysis (empty if not analysed or no people found)."""
    cached = _fresh_cache(video_id) if video_id in VIDEO_IDS else None
    r = (cached or {}).get("roster") or {}
    from integrations import players as players_mod
    if not players_mod.ENABLED or not r.get("reliable"):   # edited / crowded footage: do not show numbers users cannot trust
        return {"players": [], "teams": False}
    return r


MADE_WORDS: tuple[str, ...] = ()
EXPORT_WORDS = ("导出", "下载", "保存")
LENGTHEN_WORDS = ("延长", "加长", "拉长", "长一点", "长一些")
SHORTEN_WORDS = ("缩短", "再短", "短一点", "短一些", "压缩")
SET_DURATION_WORDS = ("剪成", "改成", "调整为", "调整到", "改为", "变成")
TRIM_WORDS = ("保留", "只留", "只要", "精简", "留下")
STRONG_GENERATE = ("生成", "提取", "剪辑", "帮我剪", "帮我做", "做一个", "做个", "来一个", "来个", "重新")
WEAK_GENERATE = ("高光", "精彩片段", "集锦", "最精彩", "挑", "选")
# "3 个 / 三段 / 两个片段" are clip counts; the bare "一个" in "做一个集锦" is not.
COUNT_PATTERN = re.compile(r"(\d+)\s*(?:个|段|片段|条)|([一二两三四五六七八九十]+)\s*(?:段|条|个(?:片段|镜头|动作|精彩))")


@dataclass(frozen=True)
class Intent:
    """What a chat message asks for, independent of any side effects.

    ``action`` is one of: help, outcome, search, generate, resize, trim,
    restyle, export, recap, need_plan, none.
    """
    action: str
    duration: float | None = None
    direction: str | None = None  # "longer" / "shorter" for resize without a number
    clip_count: int | None = None
    scoring: bool = False
    style_change: bool = False
    made_only: bool = False
    player: int | None = None
    all_players: bool = False


def parse_intent(message: str, has_plan: bool) -> Intent:
    text = (message or "").strip().lower()
    scoring = any(word in text for word in SCORING_WORDS)
    style_change = mentions_style(text)
    duration = LightweightSession._duration_from_text(text)
    count = LightweightSession._count_from_text(text)
    made_only = any(word in text for word in MADE_WORDS)
    player = player_from_text(text)
    all_players = any(word in text for word in ALL_PLAYERS_WORDS)
    common = {"duration": duration, "clip_count": count, "scoring": scoring or made_only or player is not None, "style_change": style_change,
              "made_only": made_only, "player": player, "all_players": all_players}

    def has(words: tuple[str, ...]) -> bool:
        return any(word in text for word in words)

    if not text:
        return Intent("none", **common)
    if has(HELP_WORDS):
        return Intent("help", **common)
    if OUTCOME_QUESTION.search(text):
        return Intent("outcome", **common)
    if player is not None or all_players:  # player filtering is retained as a future extension
        return Intent("generate", **common)

    longer, shorter = has(LENGTHEN_WORDS), has(SHORTEN_WORDS)
    sets_duration = has(SET_DURATION_WORDS) and duration is not None
    wants_resize = longer or shorter or sets_duration
    wants_trim = has(TRIM_WORDS) and count is not None
    strong = has(STRONG_GENERATE)

    if not has_plan:
        if wants_resize or wants_trim:
            # "缩短到 15 秒" / "保留 3 段" before any plan means: make one.
            if duration is not None or wants_trim:
                return Intent("generate", **common)
            return Intent("need_plan", **common)
    elif not strong:
        if wants_resize:
            direction = "longer" if longer else "shorter" if shorter else None
            return Intent("resize", direction=direction, **common)
        if wants_trim:
            return Intent("trim", **common)

    if has(SEARCH_WORDS):
        return Intent("search", **common)
    if strong:
        return Intent("generate", **common)
    if style_change:
        return Intent("restyle", **common)
    if has(EXPORT_WORDS):
        return Intent("export" if has_plan else "need_plan", **common)
    if has(RECAP_WORDS):
        return Intent("recap", **common)
    if has(WEAK_GENERATE) or duration is not None:
        return Intent("generate", **common)
    return Intent("none", **common)


@dataclass
class LightweightSession:
    video_id: str = ""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    candidate_version: str = "v1"
    plan: dict[str, Any] | None = None
    traces: list[dict[str, Any]] = field(default_factory=list)
    state: str = "INTENT_PARSED"
    output_dir: Path = field(init=False)
    video_duration: float = field(init=False, default=0.0)
    edit_style: dict[str, Any] = field(default_factory=lambda: normalize_edit_style(None))
    selected_events: list[dict[str, Any]] = field(default_factory=list)
    clip_count: int | None = None  # user's "保留 N 段", kept across re-edits
    progress: Any = None  # progress(done, total) during a first-time scan
    stage_hook: Any = None  # stage_hook(name) as the job moves on
    excluded: set = field(default_factory=set)        # event ids the user removed ("去掉第 2 段")
    time_range: tuple = (None, None)                  # (start, end) source seconds; None = open
    order: str | None = None                          # None/"time", "best_first", "climax"
    captions: dict = field(default_factory=dict)      # event id -> caption text
    history: list = field(default_factory=list)       # undo snapshots, newest last
    defer_render: bool = False                        # compound commands render once at the end
    player: int | None = None                         # "只看 3 号": only this numbered player's shots

    def __post_init__(self) -> None:
        if self.video_id not in VIDEO_IDS:
            raise ValueError("unknown video id")
        if self.candidate_version not in {"v1", "v2", "hypecut"}:
            raise ValueError("unsupported candidate version")
        self.session_id = str(self.session_id)
        self.output_dir = session_directory(ROOT, ROOT / "outputs" / "lightweight_agent", self.session_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.video_duration = float(_video_info(VIDEO_IDS[self.video_id])["duration"])
        except Exception:
            self.video_duration = 0.0

    def _trace(self, tool: str, result: Any) -> None:
        self.traces.append({"initiator": "CONTROLLER", "tool": tool, "result": result, "state": self.state})

    def _events(self) -> list[dict[str, Any]]:
        if self.candidate_version == "v1":
            return events(self.video_id, analyze_if_missing=True, progress=self.progress)
        from integrations.lightweight_v2 import candidates_for_variant
        return candidates_for_variant(self.video_id, self.candidate_version)

    @staticmethod
    def _score(item: dict[str, Any]) -> float:
        return float(item.get("final_ranking_score", item.get("raw_model_score", 0.0)))

    def _make_plan(self, selected: list[dict[str, Any]], target: float = 20.0, max_clips: int | None = None, style: dict[str, Any] | None = None) -> dict[str, Any]:
        if not selected:
            raise ValueError("没有满足条件的本地动作候选")
        style = normalize_edit_style(style or self.edit_style)
        self.edit_style = style
        # Keep the complete candidate pool so "改成 15 秒" can re-edit without
        # treating already-trimmed segments as the only available source.
        self.selected_events = [dict(item) for item in selected]
        source_duration = self.video_duration or (max(float(x["end_time"]) for x in selected) + 2.0)
        lo, hi = self.time_range
        filtered = [x for x in selected if x.get("event_id") not in self.excluded
                    and (lo is None or float(x["end_time"]) > lo) and (hi is None or float(x["start_time"]) < hi)]
        self.filter_note = "" if filtered else "按你的限制已经没有可用镜头，先用全部候选。"
        selected = filtered or selected
        meta = duration_meta(target, source_duration)
        target = meta["effective_duration"]
        try:
            cuts = camera_cuts(self.video_id)
        except Exception:
            cuts = None
        clips = build_montage(selected, target, source_duration, style, max_clips=max_clips, shot_cuts=cuts)
        if not clips:
            raise ValueError("没有足够的素材生成高光")
        from .edit_ops import reorder
        clips = reorder(clips, self.order)
        for clip in clips:
            text = self.captions.get(clip.get("event_id"))
            if text and not clip.get("replay"):
                clip["label"] = text
        self.plan = {
            "schema_version": "1.2",
            "source_video_id": self.video_id,
            "target_duration": round(target, 3),
            "clips": clips,
            "edit_style": style,
            "pace_label": PACE_ZH[style["pace"]],
            "transition_label": TRANSITION_ZH[style["transition"]],
            "requested_duration": meta["requested_duration"],
            "effective_duration": meta["effective_duration"],
            "source_duration": meta["source_duration"],
            "duration_clamped": meta["duration_clamped"],
            "duration_note": meta["duration_note"],
            "evidence_status": "MODEL_PREDICTED",
            "note": "本地 X3D 只识别通用网球动作，不确认比分、得分、球员或回合归属。",
        }
        self.state = "PLAN_CREATED"; self._trace("create_highlight_plan", self.plan)
        (self.output_dir / "highlight_plan.json").write_text(json.dumps(self.plan, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.plan

    @staticmethod
    def _cn_number(value: str) -> int | None:
        digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        if not value:
            return None
        if value.isdigit():
            return int(value)
        if value == "十":
            return 10
        if "十" in value:
            left, _, right = value.partition("十")
            tens = digits.get(left, 1) if left else 1
            ones = digits.get(right, 0) if right else 0
            return tens * 10 + ones
        return digits.get(value)

    @staticmethod
    def _duration_from_text(text: str) -> float | None:
        # "1分30秒", "1.5分钟", "半分钟", "一分钟"
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:分钟|分|min)\s*(?:(\d+)\s*秒)?", text, re.I)
        if match:
            seconds = float(match.group(1)) * 60 + float(match.group(2) or 0)
            if re.search(r"(?:分钟|分|min)\s*半", text, re.I):
                seconds += 30
            return seconds
        if "半分钟" in text or "半min" in text:
            return 30.0
        match = re.search(r"([零一二两三四五六七八九十]+)\s*(?:分钟|分)", text)
        if match:
            number = LightweightSession._cn_number(match.group(1))
            if number:
                seconds = float(number) * 60
                if re.search(r"(?:分钟|分)\s*半", text):
                    seconds += 30
                return seconds
        match = re.search(r"([零一二两三四五六七八九十]+)\s*秒", text)
        if match:
            number = LightweightSession._cn_number(match.group(1))
            if number:
                return float(number)
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:秒|s\b)", text, re.I)
        return float(match.group(1)) if match else None

    @staticmethod
    def _count_from_text(text: str) -> int | None:
        """Explicit clip count ("3 段", "三个片段"), or None when not stated."""
        match = COUNT_PATTERN.search(text)
        if not match:
            return None
        number = LightweightSession._cn_number(match.group(1) or match.group(2))
        return max(1, min(20, number)) if number else None

    def _write_plan(self) -> None:
        if self.plan:
            (self.output_dir / "highlight_plan.json").write_text(json.dumps(self.plan, ensure_ascii=False, indent=2), encoding="utf-8")

    def _render(self) -> dict[str, Any]:
        if not self.plan:
            raise ValueError("尚无剪辑计划")
        if self.defer_render:
            d = float(self.plan.get("target_duration", 0))
            return {"duration": d, "requested_duration": self.plan.get("requested_duration", d), "effective_duration": d,
                    "duration_note": self.plan.get("duration_note", ""), "duration_clamped": False, "deferred": True}
        if self.stage_hook:
            self.stage_hook("RENDERING")
        result = render_plan(self.video_id, self.plan, self.output_dir / "highlights.mp4")
        self.plan["rendered_window"] = {"start": result["start"], "end": result["end"], "duration": result["duration"]}
        self.plan["render_mode"] = result["render_mode"]
        self.plan["segments"] = result["segments"]
        self.plan["duration"] = result["duration"]
        self.plan["duration_clamped"] = result["duration_clamped"]
        self.plan["duration_note"] = result["duration_note"]
        self._write_plan()
        self.state = "RENDER_COMPLETED"; self._trace("render_highlights", result)
        self.state = "OUTPUT_VERIFIED"; self._trace("verify_render", result["probe"])
        return result

    def _rebuild(self, pool: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
        """Re-edit from the candidate pool, keeping the duration explanation."""
        plan = self._make_plan(pool, meta["effective_duration"], self.clip_count, style=self.edit_style)
        plan.update({key: meta[key] for key in ("requested_duration", "effective_duration", "source_duration", "duration_clamped", "duration_note") if key in meta})
        self._write_plan()
        return plan

    @staticmethod
    def _plays(plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Distinct plays; a slow-motion replay revisits one and is not counted."""
        return [clip for clip in plan.get("clips", []) if not clip.get("replay")]

    def _style_text(self, plan: dict[str, Any] | None = None) -> str:
        bits = [
            {"tight": "快节奏", "standard": "标准节奏", "relaxed": "舒缓节奏"}[self.edit_style["pace"]],
            "硬切" if self.edit_style["transition"] == "cut" else "柔和转场",
            "带字幕" if self.edit_style["captions"] else "无字幕",
        ]
        style = self.edit_style
        if style["transition"] == "flash":
            bits[1] = "闪白转场"
        if style.get("look", "vivid") != "vivid":
            bits.append(LOOK_ZH[style["look"]] + "调色")
        if style.get("zoom"):
            bits.append("推镜特效")
        if style.get("aspect") == "9:16":
            bits.append("竖屏")
        if style.get("music", "none") != "none":
            from .music import MUSIC_ZH
            bits.append(("「" + MUSIC_ZH[style["music"]] + "」配乐") if style["music"] in MUSIC_ZH else "你的配乐")
        if plan is None:
            bits.append("慢动作回放" if self.edit_style["slow_motion"] else "无慢动作")
        elif any(clip.get("replay") for clip in plan.get("clips", [])):
            bits.append("含慢动作回放")
        elif self.edit_style["slow_motion"]:
            bits.append("时长太短，没加慢动作")
        return "、".join(bits)

    # ---------- multi-step chat: undo, segments, ranges, order, compound sentences ----------
    def _snapshot(self) -> dict[str, Any]:
        import copy
        return copy.deepcopy({"plan": self.plan, "edit_style": self.edit_style, "clip_count": self.clip_count,
                              "selected_events": self.selected_events, "excluded": self.excluded,
                              "time_range": self.time_range, "order": self.order, "captions": self.captions,
                              "player": self.player})

    def _fingerprint(self) -> str:
        if not self.plan:
            return ""
        return json.dumps([[c.get("event_id"), c.get("start_time"), c.get("label"), c.get("replay")] for c in self.plan.get("clips", [])]
                          + [self.plan.get("target_duration"), self.edit_style], ensure_ascii=False, sort_keys=True, default=str)

    def _replan(self) -> dict[str, Any]:
        meta = {k: self.plan[k] for k in ("requested_duration", "effective_duration", "source_duration", "duration_clamped", "duration_note") if k in self.plan}
        meta.setdefault("effective_duration", float(self.plan["target_duration"]))
        meta.setdefault("duration_note", "")
        return self._rebuild(list(self.selected_events or self.plan.get("clips", [])), meta)

    def _apply_ops(self, ops: Any) -> str:
        """Change state for one clause's operations; returns what was done (for the reply)."""
        done: list[str] = []
        plays = self._plays(self.plan) if self.plan else []
        if ops.reset:
            self.excluded, self.time_range, self.order, self.captions = set(), (None, None), None, {}
            done.append("已清除所有限制")
        if ops.range is not None:
            self.time_range = tuple(ops.range)
            lo, hi = self.time_range
            done.append("已取消时间范围" if lo is None and hi is None else
                        f"只用原片 {_clock(lo or 0)}–{_clock(hi) if hi is not None else '结尾'} 的镜头")
        if ops.keep and plays:
            keep_ids = {plays[n - 1].get("event_id") for n in ops.keep if n <= len(plays)}
            self.excluded |= {x.get("event_id") for x in (self.selected_events or []) if x.get("event_id") not in keep_ids}
            self.clip_count = len(keep_ids)
            done.append("只保留第 " + "、".join(map(str, ops.keep)) + " 段")
        if ops.remove and plays:
            self.excluded |= {plays[n - 1].get("event_id") for n in ops.remove if n <= len(plays)}
            done.append("去掉了第 " + "、".join(map(str, ops.remove)) + " 段")
        if ops.order:
            self.order = ops.order
            done.append({"time": "按原片时间排", "best_first": "最精彩的放在最前面", "climax": "越往后越精彩，最好的一球压轴"}[ops.order])
        for n, text in ops.captions.items():
            if n <= len(plays):
                self.captions[plays[n - 1].get("event_id")] = text
                done.append(f"第 {n} 段字幕改成「{text}」")
        return "，".join(done)

    def ask(self, message: str, edit_style: dict[str, Any] | str | None = None, selected_segment: int | None = None) -> dict[str, Any]:
        """One chat message: may hold several edits; renders once; supports undo."""
        from .edit_ops import parse_ops, resolve_pronoun, split_clauses, suggestions
        text = resolve_pronoun((message or "").strip(), selected_segment)
        clauses = split_clauses(text)
        # clauses that only refine a fresh reel ("20秒", "只要进球", "竖屏") join the generate clause
        merged: list[str] = []
        for clause in clauses:
            gen = parse_intent(clause.lower(), has_plan=False).action == "generate"
            if merged and (gen or parse_intent(merged[-1].lower(), has_plan=False).action == "generate") \
                    and not parse_ops(clause, 0, float(self.video_duration or 0)).any:
                merged[-1] = merged[-1] + "，" + clause
            else:
                merged.append(clause)
        clauses = merged
        before = self._snapshot() if self.plan else None
        before_fp = self._fingerprint()
        replies: list[str] = []
        last: dict[str, Any] = {}
        changed = False
        self.ops_set = False
        try:
            for index, clause in enumerate(clauses):
                final = index == len(clauses) - 1
                plays = len(self._plays(self.plan)) if self.plan else 0
                ops = parse_ops(clause, plays, float(self.video_duration or 0))
                if ops.undo:
                    if not self.history:
                        replies.append("没有可以撤销的修改了。"); continue
                    snap = self.history.pop()
                    for key, value in snap.items():
                        setattr(self, key, value)
                    self._write_plan()
                    self.defer_render = False
                    output = self._render() if self.plan else None
                    replies.append("已撤销上一步修改。"); last = {"output": output, "plan": self.plan} if output else {}
                    before, before_fp = None, self._fingerprint()     # undo itself is not pushed to history
                    continue
                self.defer_render = not final
                if ops.any:
                    self.ops_set = True
                    done = self._apply_ops(ops)
                    if self.plan and not ops.rest:
                        self._replan()
                        output = self._render()
                        last = {"output": output, "plan": self.plan}
                        note = (" " + self.filter_note) if getattr(self, "filter_note", "") else ""
                        replies.append((done or "已更新") + "。" + note); changed = True
                        continue
                    if done and not self.plan and not ops.rest:
                        replies.append(done + "，生成集锦时会按这个来。"); continue
                    if done:
                        replies.append(done + "。")
                    clause = ops.rest or clause
                result = self._ask_one(clause, edit_style if index == 0 else None)
                replies.append(result.get("reply", ""))
                last = {k: v for k, v in result.items() if k not in ("reply", "suggestions")}
                changed = changed or bool(result.get("output"))
        finally:
            self.defer_render = False
        if len(clauses) > 1 and self.plan and last.get("output", {}).get("deferred"):
            last["output"] = self._render()          # the one real render for a compound message
        if before is not None and self._fingerprint() != before_fp:
            self.history = (self.history + [before])[-10:]
        reply = " ".join(r for r in replies if r).strip() or "好的。"
        if len(clauses) > 1 and last.get("output"):
            reply += f"（成片 {float(last['output']['duration']):.1f} 秒）"
        out = self._save(reply, last)
        out["suggestions"] = suggestions(self.plan, any(x.get("outcome") == "made" for x in (self.selected_events or [])),
                                         can_undo=bool(self.history))
        out["can_undo"] = bool(self.history)
        return out

    def _ask_one(self, message: str, edit_style: dict[str, Any] | str | None = None) -> dict[str, Any]:
        """Execute a supported local command using state, rather than canned answers."""
        text = message.strip().lower()
        if not text:
            return self._save("请输入一条命令，例如“生成 20 秒网球高光”。", {})
        try:
            if edit_style is not None:
                self.edit_style = normalize_edit_style(edit_style)
            # Chat can override the UI's style ("做成紧凑一点 / 加慢放").
            self.edit_style = edit_style_from_text(message, self.edit_style)
            is_hypecut = self.candidate_version == "hypecut"
            intent = parse_intent(text, has_plan=bool(self.plan))
            duration = intent.duration
            player_note = ""
            if intent.all_players:
                self.player = None
                player_note = " 已改回所有人的集锦。"
            elif intent.player is not None and not is_hypecut:
                return self._save("当前网球版暂不提供按球员筛选；可以按时间范围、节奏或片段数来剪。", {})

            if intent.action == "help":
                return self._save(
                    "我可以帮你：\n"
                    "• 生成 20 秒网球高光\n"
                    "• 做一个 30 秒精彩集锦\n"
                    "• 找出所有疑似发球/击球\n"
                    "• 缩短到 15 秒 / 延长到 30 秒 / 只保留 3 段\n"
                    "• 改成硬切 / 加慢放回看 / 不要字幕\n"
                    "• 加热血配乐 / 换一首 / 电影感 / 复古 / 黑白 / 加特效\n"
                    "• 标题改成“球场之夜”\n"
                    "• 去掉第 2 段 / 只保留第 1、3 段 / 第 2 段字幕改成“制胜回合”\n"
                    "• 只要后半段 / 1:30 以后 / 最后 2 分钟\n"
                    "• 最精彩的放最后 / 按时间顺序 / 撤销\n"
                    "• 一句话说好几件事：“去掉第 2 段，节奏快一点，改成竖屏”\n"
                    "• 分析一下这场 / 找出发球镜头 / 导出视频\n"
                    "直接说你想做什么即可。",
                    {},
                )
            if intent.action == "restyle" and not self.plan:
                return self._save(
                    f"已记下剪辑风格（{self._style_text()}），生成高光时会使用。试试说“生成 20 秒网球高光”。",
                    {"edit_style": self.edit_style},
                )
            if intent.action == "need_plan":
                return self._save("当前还没有高光计划。先告诉我“生成 20 秒高光”，之后才能继续调整。", {"failure": {"stage": "STATE_CHECK", "retryable": True}})
            if intent.action == "none":
                # Do not run a long local scan just to answer an unrelated
                # message.  Give a useful next step instead.
                from .edit_ops import did_you_mean
                close = did_you_mean(message)
                hint = ("你是想说：" + " / ".join(f"“{x}”" for x in close) + " 吗？") if close else \
                    "试试说：“生成 20 秒网球高光”“找出发球镜头”“去掉第 2 段”“只要后半段”。"
                return self._save(
                    "这句我没听懂。" + hint + "\n一句话里可以说好几件事，比如“去掉第 2 段，节奏快一点，最精彩的放最后”。",
                    {},
                )

            def clamp_meta(value: float) -> dict[str, Any]:
                # 60 s is the product cap, not a physical limit.  The source
                # duration is the real ceiling and is explained to the user.
                requested = max(5.0, min(60.0, float(value)))
                return duration_meta(requested, self.video_duration or requested)

            # Re-edits of an existing plan reuse its candidate pool; they do
            # not need a fresh scan.
            if intent.action in {"resize", "trim", "restyle"}:
                current = float(self.plan["target_duration"])
                if intent.action == "resize":
                    if duration is None:
                        duration = current + 5.0 if intent.direction == "longer" else current - 5.0
                    meta = clamp_meta(duration)
                    if abs(meta["effective_duration"] - current) < 0.01:
                        note = (" " + meta["duration_note"]) if meta["duration_note"] else ""
                        return self._save(f"当前高光已经是约 {current:g} 秒，无需调整。{note}".strip(), {"plan": self.plan})
                else:
                    meta = {key: self.plan[key] for key in ("requested_duration", "effective_duration", "source_duration", "duration_clamped", "duration_note") if key in self.plan}
                    meta.setdefault("effective_duration", current)
                    meta.setdefault("duration_note", "")
                if intent.action == "trim":
                    self.clip_count = intent.clip_count
                pool = list(self.selected_events or self.plan.get("clips", []))
                plan = self._rebuild(pool, meta)
                self._trace("update_highlight_plan", {"action": intent.action, "target_duration": plan["target_duration"],
                                                      "previous_duration": current, "clip_count": self.clip_count, "edit_style": self.edit_style})
                output = self._render()
                plays = self._plays(plan)
                if intent.action == "resize":
                    head = f"已把高光调整为 {output['duration']:.1f} 秒，共 {len(plays)} 个片段"
                elif intent.action == "trim":
                    head = f"已精简为 {len(plays)} 个片段，时长 {output['duration']:.1f} 秒"
                    if self.clip_count and len(plays) < self.clip_count:
                        head += f"（素材只够 {len(plays)} 段）"
                else:
                    head = f"已按新风格重新剪辑：{output['duration']:.1f} 秒，共 {len(plays)} 个片段"
                note = (" " + meta["duration_note"]) if intent.action == "resize" and meta.get("duration_note") else ""
                return self._save(f"{head}（{self._style_text(plan)}）。{note}".strip(), {"output": output, "plan": self.plan})

            if intent.action == "export":
                output = self._render()
                return self._save("已重新导出当前高光，视频可以直接播放或下载。", {"output": output, "plan": self.plan})

            items = self._events()
            self._trace("get_video_info", {"video_id": self.video_id, "candidate_version": self.candidate_version, "events_available": len(items)})

            def matching(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
                if is_hypecut or not intent.scoring:
                    return list(pool)
                return [x for x in pool if x["event_type"] in SHOT_LIKE_TYPES]

            if intent.action == "outcome":
                self.state = "EVENTS_RETRIEVED"
                self._trace("search_events", {"event_type": "tennis_action_candidate", "count": len(items), "outcome_check": True})
                return self._save("当前版本只定位疑似发球和击球镜头，不判断得分、比分、胜负或回合结果。请以原始比赛画面和记分牌为准。", {"events": matching(items)})

            if intent.action == "search":
                found = matching(items)
                self.state = "EVENTS_RETRIEVED"
                self._trace("search_events", {"event_type": "tennis_hit" if intent.scoring and not is_hypecut else None, "count": len(found)})
                if not found:
                    return self._save("没找到符合条件的镜头，可以试试“找出所有精彩动作”。", {"events": []})
                ranked = sorted(found, key=self._score, reverse=True)[:3]
                preview = "；".join(
                    f"{float(x['start_time']):.1f}s {x.get('display_name') or EVENT_TYPE_ZH.get(x['event_type'], x['event_type'])}"
                    for x in ranked
                )
                kind = "发球/击球" if intent.scoring and not is_hypecut else "网球动作"
                return self._save(
                    f"找到 {len(found)} 个{kind}镜头，最值得先看：{preview}。这些是自动识别的，可能有误差。",
                    {"events": found},
                )

            if intent.action == "recap":
                from . import tennis_review as review_mod
                report = review_mod.build(self.video_id, items, float(self.video_duration or 0) or max(
                    [float(x["end_time"]) for x in items] or [1.0]))
                if report["candidate_count"]:
                    lines = list(report["notes"])
                    busy = [p for p in report["periods"] if p["count"]]
                    if len(busy) > 1:
                        lines.append("分时段：" + "；".join(
                            f"{p['label']} 候选 {p['count']} 个" for p in busy))
                    summary = " ".join(lines) + " 点「📊 网球复盘」可以看时间线和重点时段。结果是自动识别的，可能有误。"
                    recap = {"source_video_id": self.video_id, "verification_status": "MODEL_PREDICTED", "summary": summary,
                             "candidate_count": report["candidate_count"], "periods": report["periods"]}
                    self._trace("generate_recap", recap)
                    return self._save(summary, {"recap": recap})
                if self.plan and self.plan.get("clips"):
                    lines = []
                    for index, clip in enumerate(self.plan["clips"], 1):
                        name = clip.get("label") or clip.get("display_name") or EVENT_TYPE_ZH.get(clip.get("event_type", ""), "动作")
                        out = clip.get("output_duration") or (float(clip["end_time"]) - float(clip["start_time"]))
                        lines.append(f"{index}. {name}，原片 {float(clip['start_time']):.1f}–{float(clip['end_time']):.1f}s，成片约 {float(out):.1f}s")
                    summary = "当前集锦包含：" + "；".join(lines) + "。动作是自动识别的，得分、比分和球员信息需要你回看确认。"
                else:
                    top = sorted(items, key=self._score, reverse=True)[:5]
                    summary = (
                        "这段视频最值得看的是："
                        + "；".join(
                            f"{float(x['start_time']):.1f}s {x.get('display_name') or EVENT_TYPE_ZH.get(x['event_type'], x['event_type'])}"
                            for x in top
                        )
                        + "。先生成高光可以进一步筛选。"
                    ) if top else "这段视频里暂时没认出网球动作。"
                recap = {"source_video_id": self.video_id, "events": self.plan["clips"] if self.plan else [],
                         "verification_status": "MODEL_PREDICTED", "summary": summary}
                self._trace("generate_recap", recap)
                (self.output_dir / "match_recap.json").write_text(json.dumps(recap, ensure_ascii=False, indent=2), encoding="utf-8")
                return self._save(summary, {"recap": recap})

            # intent.action == "generate"
            if not getattr(self, "ops_set", False):
                self.excluded, self.time_range, self.order, self.captions = set(), (None, None), None, {}
            selected = matching(items)
            fallback_note = ""
            if not selected:
                if not items:
                    return self._save("这段视频里暂时没认出网球动作，生成不了集锦，换一段机位更稳定的网球视频试试吧。", {"failure": {"stage": "NO_CANDIDATES", "retryable": True}})
                selected = list(items)
                fallback_note = " 发球或击球候选不够多，已经用上了所有精彩动作。"
            # A fresh request resets "保留 N 段" unless it states a count itself.
            self.clip_count = intent.clip_count
            meta = clamp_meta(duration if duration is not None else 20.0)
            plan = self._rebuild(selected, meta)
            output = self._render()
            plays = self._plays(plan)
            counts = Counter(c.get("label") or c.get("display_name") or EVENT_TYPE_ZH.get(c.get("event_type", ""), "动作") for c in plays)
            summary = "、".join(f"{name} {count} 段" for name, count in counts.items())
            note = (" " + meta["duration_note"]) if meta["duration_note"] else ""
            if is_hypecut:
                reply = (
                    f"已按画面变化剪好 {output['duration']:.1f} 秒集锦，共 {len(plays)} 个片段"
                    f"（{self._style_text(plan)}）。"
                    "这种方式挑的是镜头最热闹的地方，不一定是网球动作。"
                )
            else:
                reply = (
                    f"高光已生成：{output['duration']:.1f} 秒，共 {len(plays)} 个片段"
                    + (f"（{summary}）" if summary else "")
                    + f"。剪辑风格：{self._style_text(plan)}。"
                )
            return self._save(reply + note + fallback_note + player_note, {"plan": plan, "output": output})
        except Exception as exc:
            failure = {"stage": self.state, "error_type": type(exc).__name__, "message": str(exc), "retryable": isinstance(exc, (RuntimeError, OSError, subprocess.SubprocessError))}
            self._trace("agent_failure", failure)
            return self._save("这次没剪成，" + ("请稍后重试。" if failure["retryable"] else "换个说法再试试？"), {"failure": failure})

    def _save(self, reply: str, extra: dict[str, Any]) -> dict[str, Any]:
        if not extra.get("failure"):
            self.state = "TASK_COMPLETED" if (self.output_dir / "highlights.mp4").exists() else self.state
        (self.output_dir / "tool_trace.json").write_text(json.dumps(self.traces, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "session_history.json").write_text(json.dumps({"video_id": self.video_id, "state": self.state}, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "run_summary.json").write_text(json.dumps({"mode": "lightweight", "candidate_version": self.candidate_version, "state": self.state, "tool_calls": len(self.traces), "cloud_api_calls": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
        # edit_style is the style now in effect (UI + chat overrides); the page
        # adopts it so the next message does not send a stale style back.
        return {"reply": reply, "session_id": self.session_id, "mode": "lightweight", "candidate_version": self.candidate_version,
                "artifacts": str(self.output_dir.relative_to(ROOT)), "state": self.state, "edit_style": dict(self.edit_style), **extra}
