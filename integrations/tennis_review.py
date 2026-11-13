"""Evidence-bounded tennis review built from highlight candidates."""
from __future__ import annotations

from typing import Any


def _clock(value: float) -> str:
    return f"{int(value // 60)}:{int(value % 60):02d}"


def build(video_id: str, events: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    ordered = sorted(events, key=lambda event: float(event.get("raw_model_score", 0)), reverse=True)
    moments = [{"time": round((float(event["start_time"]) + float(event["end_time"])) / 2, 3),
                "clock": _clock((float(event["start_time"]) + float(event["end_time"])) / 2),
                "label": event.get("display_name") or "精彩回合候选",
                "score": round(float(event.get("raw_model_score", 0)), 3)} for event in ordered]
    bucket_size = 30.0 if duration <= 300 else 60.0
    buckets: dict[int, int] = {}
    for moment in moments:
        buckets[int(moment["time"] // bucket_size)] = buckets.get(int(moment["time"] // bucket_size), 0) + 1
    active = sorted(buckets, key=lambda index: buckets[index], reverse=True)[:3]
    active_periods = [{"label": f"{_clock(index * bucket_size)}–{_clock(min(duration, (index + 1) * bucket_size))}", "count": buckets[index]} for index in active]
    notes = [f"自动筛出 {len(moments)} 个值得回看的网球动作候选。"]
    if active_periods:
        notes.append(f"候选最密集的时段是 {active_periods[0]['label']}（{active_periods[0]['count']} 个候选）。")
    notes.append("候选依据为模型或场地运动强度；不代表得分、胜负、发球成功或回合结果。")
    return {"kind": "tennis_review", "video_id": video_id, "duration": duration, "candidate_count": len(moments),
            "moments": moments, "active_periods": active_periods, "notes": notes,
            "caveat": "请以原始比赛画面和记分牌为准。"}
