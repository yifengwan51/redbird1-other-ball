"""Evidence-bounded post-match review for tennis highlight candidates."""
from __future__ import annotations

import html
from typing import Any


def _clock(value: float) -> str:
    value = max(0, int(round(value)))
    return f"{value // 60}:{value % 60:02d}"


def build(video_id: str, events: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    """Build a useful review without inventing points, score, or winners."""
    ordered = sorted(events, key=lambda event: float(event.get("final_ranking_score", event.get("raw_model_score", 0))), reverse=True)
    moments = []
    for rank, event in enumerate(ordered, 1):
        start, end = float(event["start_time"]), float(event["end_time"])
        time = round((start + end) / 2, 3)
        moments.append({"rank": rank, "time": time, "peak": time, "clock": _clock(time),
                        "label": event.get("display_name") or "精彩回合候选",
                        "score": round(float(event.get("final_ranking_score", event.get("raw_model_score", 0))), 3),
                        "reason": event.get("description") or "动作与场地运动强度较高"})
    bucket_size = 30.0 if duration <= 300 else 60.0
    buckets: dict[int, list[dict[str, Any]]] = {}
    for moment in moments:
        buckets.setdefault(int(moment["time"] // bucket_size), []).append(moment)
    periods = []
    for index in range(max(1, int((duration + bucket_size - 0.01) // bucket_size))):
        inside = buckets.get(index, [])
        periods.append({"label": f"{_clock(index * bucket_size)}–{_clock(min(duration, (index + 1) * bucket_size))}",
                        "start": index * bucket_size, "end": min(duration, (index + 1) * bucket_size), "count": len(inside),
                        "avg_score": round(sum(x["score"] for x in inside) / len(inside), 3) if inside else None})
    active_periods = sorted((p for p in periods if p["count"]), key=lambda p: (p["count"], p["avg_score"] or 0), reverse=True)[:3]
    best = moments[:3]
    notes = [f"自动筛出 {len(moments)} 个值得回看的网球动作候选。"]
    if active_periods:
        notes.append(f"候选最密集的时段是 {active_periods[0]['label']}（{active_periods[0]['count']} 个候选）。")
    if best:
        notes.append(f"优先回看 {_clock(best[0]['time'])} 附近的镜头；它在当前候选排序中最靠前。")
    notes.append("候选依据为画面动作与场地运动强度；不代表得分、胜负、发球成功或回合结果。")
    return {"kind": "tennis_review", "video_id": video_id, "duration": duration, "candidate_count": len(moments),
            "moments": moments, "periods": periods, "active_periods": active_periods, "best": best, "notes": notes,
            "caveat": "请以原始比赛画面和记分牌为准。"}


RECAP_PROMPT = """你是网球教练助理。根据以下自动生成的候选复盘，写 3–4 句中文赛后复盘。
只能谈值得回看的原片时段、剪辑节奏和观察建议；不要编造比分、胜负、球员、发球成功或回合结果。语气像教练，简洁具体。

{data}"""


def recap_input(report: dict[str, Any]) -> str:
    periods = [p for p in report["periods"] if p["count"]]
    return (f"视频 {_clock(report['duration'])}；回看候选 {report['candidate_count']} 个。\n重点时段："
            + "；".join(f"{p['label']} {p['count']} 个候选" for p in periods[:6]) + "。\n优先回看："
            + "、".join(f"{m['clock']} {m['label']}" for m in report["best"]) + "。")


def recap_prompt(report: dict[str, Any]) -> str:
    return RECAP_PROMPT.format(data=recap_input(report))


def to_html(report: dict[str, Any], title: str, recap: str | None = None) -> str:
    e = html.escape
    best = "".join(f"<li>{e(m['clock'])} · {e(m['label'])}</li>" for m in report["best"])
    periods = "".join(f"<tr><td>{e(p['label'])}</td><td>{p['count']}</td><td>{p['avg_score'] if p['avg_score'] is not None else '—'}</td></tr>" for p in report["periods"])
    notes = "".join(f"<li>{e(n)}</li>" for n in report["notes"])
    return f'''<!doctype html><meta charset="utf-8"><title>{e(title)} · 网球复盘</title>
<style>body{{max-width:780px;margin:32px auto;padding:0 16px;font:15px/1.65 system-ui;color:#182033}}.tiles{{display:flex;gap:12px}}.tile,section{{border:1px solid #dce1ea;border-radius:12px;padding:14px;margin:16px 0}}.tile b{{font-size:26px;display:block}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #e6e9ef;text-align:left}}small{{color:#667085}}</style>
<h1>{e(title)} · 网球复盘</h1><small>视频 {_clock(report['duration'])} · 自动生成</small>
<div class="tiles"><div class="tile"><b>{report['candidate_count']}</b>回看候选</div><div class="tile"><b>{_clock(report['duration'])}</b>视频时长</div></div>
{f'<section><h2>AI 复盘</h2><p>{e(recap)}</p></section>' if recap else ''}
<section><h2>复盘要点</h2><ul>{notes}</ul></section><section><h2>优先回看</h2><ul>{best}</ul></section>
<section><h2>分时段候选</h2><table><tr><th>时段</th><th>候选</th><th>平均强度</th></tr>{periods}</table></section><small>{e(report['caveat'])}</small>'''
