"""VIP "AI 剪辑师": a real LLM decides the edit, the local editor executes it.

The model never touches files or FFmpeg.  It sees a compact list of candidate
moments (local X3D candidates plus cached vision evidence) and the current
plan, and may call exactly one tool, ``edit_highlight``, whose validated
arguments drive the same deterministic montage renderer as the free editor.
That keeps durations exact and styles consistent between the two tiers.

Cost bounds: at most two model requests per turn (decide + reply), a short
rolling history, and a per-session request budget.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import uuid
from typing import Any

import cv2
import numpy as np

from .lightweight import (
    ROOT, SHOT_LIKE_TYPES, VIDEO_IDS, EVENT_TYPE_ZH, PACE_ZH, TRANSITION_ZH,
    LightweightSession, duration_meta, normalize_edit_style, _json, _fingerprint,
)

VISION_CACHE = ROOT / "outputs" / "semantic_vision_live" / "tennis_events.json"
KEY_FILE = ROOT / ".secrets" / "htadc_api_key"
DEFAULT_MODEL = "deepseek-v4.1-flash"
MAX_REQUESTS_PER_SESSION = 80
HISTORY_MESSAGES = 8          # last 4 user/assistant exchanges
INCLUDE_BOOST = 10.0          # ranks pinned moments above every model score


class BudgetExceeded(RuntimeError):
    pass


def api_key() -> str:
    key = os.getenv("MOTRACK_API_KEY") or (KEY_FILE.read_text(encoding="utf-8").strip() if KEY_FILE.is_file() else "")
    if not key:
        raise RuntimeError("AI 剪辑服务未配置")
    return key


def vision_events(video_id: str) -> list[dict[str, Any]]:
    """Cached vision-model evidence for one video, in the local event shape."""
    payload = _json(VISION_CACHE, {})
    items = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    name = VIDEO_IDS[video_id].name
    out = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not str(item.get("video_path", "")).endswith(name):
            continue
        try:
            start = float(item["start_time"])
            end = max(float(item.get("end_time", start)), start + 1.6)
        except (KeyError, TypeError, ValueError):
            continue
        kind = item.get("event_type") if item.get("event_type") in EVENT_TYPE_ZH else "tennis_action_candidate"
        out.append({
            "event_id": f"vision-{video_id}-{index + 1}",
            "source_video_id": video_id,
            "event_type": kind,
            "display_name": EVENT_TYPE_ZH.get(kind, "网球动作"),
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "raw_model_score": float(item.get("confidence", 0.5)),
            "description": str(item.get("evidence_description", ""))[:220],
            "evidence_source": "CACHED_VISION",
            "verification_status": "MODEL_PREDICTED",
        })
    return out


class VisionAwareSession(LightweightSession):
    """Local candidates plus cached vision evidence, ranked together."""

    def _events(self) -> list[dict[str, Any]]:
        return super()._events() + vision_events(self.video_id)


SYSTEM_PROMPT = """你是 MoTrack 的 AI 剪辑师，帮球迷把网球视频剪成高光集锦。

说话方式：
- 用自然、简短、热情的中文，像一位懂球的剪辑师，一般 1–3 句话。
- 不要提模型、分数、置信度、id、JSON、参数名等技术词；提到片段时说"第 2 段"或"原片 1:17 那一球"。

怎么做事：
- 用户想生成或修改集锦时，调用 edit_highlight，只填这次要改的字段；之前去掉或保留的选择会自动沿用，不要重复填。
- 常见说法："去掉第 2 段"→ remove_segments=[2]；"一定要有 1:17 那球"→ pin_moments 填那个时刻；"恢复原样"→ reset_choices=true；"只要后半段"→ start_after；"3 段就够"→ clip_count；"节奏快点"→ pace=tight；"加个慢动作"→ slow_motion=true；"来点燃的音乐"→ music=hype；"做出电影感"→ look=cinematic；"加点特效"→ zoom=true 并 transition=flash。
- 需要确认画面内容时（“是不是精彩回合”“哪一拍最精彩”“去掉没看点的”），先调用 inspect_moments 看截图，再回答或剪辑；标了“AI看过”的不用再看。
- 一句话里有好几个要求时（“去掉第 2 段，节奏快点，最精彩的放最后”），一次 edit_highlight 把所有字段都填上，不要分几次剪。
- 顺序：“最精彩的放最后/压轴”→ order=climax；“精彩的先放”→ order=best_first；“按时间顺序”→ order=time。
- 改某段字幕：segment_captions={{"2":"压哨三分"}}。用户说“这段/那段”且消息开头写了“我点选了第 N 段”，就指第 N 段。
- 用户说“撤销”“改回去”“还是刚才的好”→ 调用 undo_edit。
- 问比赛表现、哪段时间最值得回看时，先调用 get_game_stats 拿候选数据再回答，不要凭印象。
- 剪完后用一句话说清改了什么、现在多长，再给一个具体的下一步建议。
- 只是聊天、提问或让你解释时，直接回答，不要调用工具。
- 配乐：内置 hype（热血电子）、groove（嘻哈律动）、chill（轻快）三首，用了内置配乐会自动卡点剪辑；用户上传过的音乐可以用 upload:xxxx。指定某首具体的歌（比如周杰伦的歌）做不到，请用户自己上传。
- 特效：调色 look、推镜 zoom、闪白转场 transition=flash、片头标题 title。
- 想要“发球集锦”时，调用 edit_highlight 设 focus=shots（发球/击球候选）；想要所有动作用 focus=all。
- 本系统不判断比分、胜负、发球成功或回合结果；被问这些时明确说明需要回看原片和记分牌。
- 做不到的（按球员或球队筛选、换背景、加贴纸）要坦白说明，并给出最接近的做法。

事实边界：
- 候选时刻来自动作与画面运动强度；它们不等于得分或获胜回合。
- 比分、胜负、球员、发球是否成功和回合结果都不能识别，绝不编造。
- 有画面描述的时刻可以引用描述，但要说"看起来"。
- 集锦最长 60 秒，也不能超过原片长度。

视频：{video_name}，原片 {duration} 秒。
候选复盘：{shooting}
候选时刻（时间、动作、强度、描述）：
{moments}

当前集锦：
{plan}"""

COMMENTARY_PROMPT = """你是网球集锦解说。根据下面的片段清单，写 2–3 句中文解说，介绍这段集锦的看点，语气有现场感。
不要编造比分、球员、胜负、发球成功或回合结果；不确定的动作说“疑似”或“看起来”。不要提任何技术细节。"""

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_highlight",
        "description": "生成或修改当前集锦。只传需要改变的字段；未传的字段保持现状。",
        "parameters": {
            "type": "object",
            "properties": {
                "duration": {"type": "number", "description": "成片秒数，5–60"},
                "focus": {"type": "string", "enum": ["shots", "all"], "description": "shots=发球/击球候选，all=所有网球动作候选"},
                "clip_count": {"type": "integer", "description": "最多几段；0 表示自动"},
                "remove_segments": {"type": "array", "items": {"type": "integer"}, "description": "从当前集锦去掉第几段（按当前编号，从 1 开始）"},
                "remove_moments": {"type": "array", "items": {"type": "string"}, "description": "不要用的候选时刻 id，如 m08"},
                "pin_moments": {"type": "array", "items": {"type": "string"}, "description": "一定要放进集锦的候选时刻 id"},
                "reset_choices": {"type": "boolean", "description": "true=撤销之前所有去掉/保留/范围限制"},
                "start_after": {"type": ["number", "null"], "description": "只用这个原片秒数之后的内容；null 取消"},
                "end_before": {"type": ["number", "null"], "description": "只用这个原片秒数之前的内容；null 取消"},
                "pace": {"type": "string", "enum": ["tight", "standard", "relaxed"]},
                "transition": {"type": "string", "enum": ["fade", "cut", "flash"], "description": "fade=柔和，cut=硬切，flash=闪白"},
                "look": {"type": "string", "enum": ["vivid", "cinematic", "retro", "bw"], "description": "调色：鲜艳/电影感（带宽银幕黑边）/复古/黑白"},
                "zoom": {"type": "boolean", "description": "推镜特效：每段镜头缓缓推进"},
                "aspect": {"type": "string", "enum": ["16:9", "9:16"], "description": "画幅：16:9 横屏；9:16 竖屏（抖音/视频号），会自动跟着画面主体裁切"},
                "music": {"type": "string", "description": "配乐：none=原声，hype=热血电子，groove=嘻哈律动，chill=轻快；或用户上传的 upload:xxxx"},
                "title": {"type": ["string", "null"], "description": "片头标题，最多 20 字；\"\"=不要标题；null=默认"},
                "captions": {"type": "boolean"},
                "order": {"type": "string", "enum": ["time", "best_first", "climax"],
                          "description": "片段顺序：time=按原片时间，best_first=最精彩的放最前，climax=越往后越精彩、最好的压轴"},
                "segment_captions": {"type": "object", "additionalProperties": {"type": "string"},
                                     "description": "改某几段的字幕，键是当前段号（\"2\"），值是新字幕（最多 16 字）"},
                "slow_motion": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
}
# ---------- AI frame review ----------
INSPECT_DIR = ROOT / "outputs" / "ai_inspect"
INSPECT_BATCH = 4            # moments per contact sheet (one image per request)
MAX_TOOL_ROUNDS = 5
ACTIONS = ("发球", "正手击球", "反手击球", "网前截击", "底线回合", "庆祝", "观众", "回放", "广告", "其他")
SHOT_ACTIONS = {"发球", "正手击球", "反手击球", "网前截击", "底线回合"}
SKIP_ACTIONS = {"观众", "回放", "广告"}
INSPECT_PROMPT = """你在审看网球比赛视频的截图。图里每一行是一个候选时刻，左上角是它的编号（如 m07），同一行从左到右是这个时刻前、中、后三帧。
只返回 JSON：{"moments":[{"id":"m07","action":"发球|正手击球|反手击球|网前截击|底线回合|庆祝|观众|回放|广告|其他","highlight":1到5的整数,"note":"不超过 20 字的画面描述"}]}
highlight：5=动作清晰且值得回看，3=普通回合，1=没有看点。看不清就照实写，不要猜比分、球员姓名、胜负或回合结果。"""
_inspect_lock = threading.Lock()


def _moment_key(item: dict[str, Any]) -> str:
    return f"{float(item['start_time']):.1f}-{float(item['end_time']):.1f}"


def _load_inspections(video_id: str) -> dict[str, dict[str, Any]]:
    data = _json(INSPECT_DIR / f"{video_id}.json", {})
    if data.get("fingerprint") != _fingerprint(VIDEO_IDS[video_id]):
        return {}
    return dict(data.get("moments", {}))


def _store_inspections(video_id: str, found: dict[str, dict[str, Any]]) -> None:
    with _inspect_lock:
        current = _load_inspections(video_id)
        current.update(found)
        INSPECT_DIR.mkdir(parents=True, exist_ok=True)
        path = INSPECT_DIR / f"{video_id}.json"
        part = path.with_name(f".{uuid.uuid4().hex[:6]}_{path.name}")
        part.write_text(json.dumps({"fingerprint": _fingerprint(VIDEO_IDS[video_id]), "moments": current},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(part, path)


def annotate(video_id: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach cached AI frame reviews (``ai``) to candidate moments."""
    seen = _load_inspections(video_id)
    return [dict(item, ai=seen[_moment_key(item)]) if _moment_key(item) in seen else item for item in items]


def _contact_sheet(video: Any, rows: list[tuple[str, float, float]]) -> str:
    """One JPEG: a row of before / middle / after frames per moment, labelled."""
    cap = cv2.VideoCapture(str(video))
    tiles = []
    try:
        for moment_id, start, end in rows:
            frames = []
            for t in (max(0.0, start - 0.4), (start + end) / 2, end + 0.4):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                frames.append(cv2.resize(frame, (256, 144)) if ok else np.zeros((144, 256, 3), np.uint8))
            row = np.hstack(frames)
            cv2.rectangle(row, (0, 0), (70, 30), (0, 0, 0), -1)
            cv2.putText(row, moment_id, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            row[-2:] = 0
            tiles.append(row)
    finally:
        cap.release()
    ok, buf = cv2.imencode(".jpg", np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("could not encode contact sheet")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _parse_inspection(text: str, allowed: set[str]) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(text or "")
    except ValueError:
        match = re.search(r"\{.*\}", text or "", re.S)
        try:
            data = json.loads(match.group(0)) if match else {}
        except ValueError:
            data = {}
    rows = data.get("moments") if isinstance(data, dict) else None
    out: dict[str, dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or str(row.get("id")) not in allowed:
            continue
        action = str(row.get("action", "其他"))
        action = action if action in ACTIONS else "其他"
        try:
            highlight = max(1, min(5, int(row.get("highlight", 3))))
        except (TypeError, ValueError):
            highlight = 3
        out[str(row["id"])] = {"action": action, "highlight": highlight, "tennis": action not in SKIP_ACTIONS,
                               "note": re.sub(r"\s+", " ", str(row.get("note", "")))[:40]}
    return out


def _review_label(review: dict[str, Any]) -> str:
    if review["action"] == "庆祝":
        return "庆祝时刻"
    return "疑似" + review["action"] if review["action"] not in ("其他",) else "网球动作"


INSPECT_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_moments",
        "description": "查看候选时刻的真实截图，确认是什么动作、有多精彩。已标“AI看过”的不用再看。",
        "parameters": {
            "type": "object",
            "properties": {"moments": {"type": "array", "items": {"type": "string"}, "description": "时刻 id，如 m07，一次最多 8 个"}},
            "required": ["moments"],
            "additionalProperties": False,
        },
    },
}


STATS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_game_stats",
        "description": "取整场的投篮统计：出手、进球、命中率、分时段、连续命中/没进、每次出手的时间和结果。回答比赛表现类问题前先调用。",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}
UNDO_TOOL = {
    "type": "function",
    "function": {
        "name": "undo_edit",
        "description": "撤销上一次 edit_highlight 的修改，恢复到之前的集锦。用户说“撤销”“改回去”“刚才那样好”时调用。",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


def _clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


class LLMEditor:
    def __init__(self, video_id: str, session_id: str, video_name: str = "", client: Any = None):
        self.session = VisionAwareSession(video_id=video_id, session_id=session_id, candidate_version="v1")
        self.video_id = video_id
        self.session_id = self.session.session_id
        self.video_name = video_name or VIDEO_IDS[video_id].stem
        self.model = os.getenv("MOTRACK_API_MODEL", DEFAULT_MODEL)
        self.settings: dict[str, Any] = {"duration": 20.0, "focus": "shots", "clip_count": 0,
                                         "include": [], "exclude": [], "start_after": None, "end_before": None}
        self.history: list[dict[str, str]] = []
        self.usage: list[dict[str, Any]] = []
        self.commentary = ""
        self._client = client
        self._usage_lock = threading.Lock()
        self._pool: list[dict[str, Any]] | None = None

    # ---------- candidate moments ----------
    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            from .api_config import resolve_base_url
            self._client = OpenAI(api_key=api_key(), base_url=resolve_base_url(), timeout=45, max_retries=1)
        return self._client

    @property
    def pool(self) -> list[dict[str, Any]]:
        """Every candidate with a short stable id (m01, m02, ...) for the model."""
        if self._pool is None:
            items = sorted(self.session._events(), key=lambda x: (float(x["start_time"]), x.get("event_id", "")))
            items = annotate(self.video_id, items)
            self._pool = [dict(item, moment_id=f"m{i + 1:02d}") for i, item in enumerate(items)]
        return self._pool

    def _by_moment(self) -> dict[str, dict[str, Any]]:
        return {item["moment_id"]: item for item in self.pool}

    def _moment_of(self, event_id: Any) -> str | None:
        return next((m["moment_id"] for m in self.pool if m.get("event_id") == event_id), None)

    @staticmethod
    def _strength(item: dict[str, Any]) -> str:
        score = float(item.get("raw_model_score", 0.0))
        return "强" if score >= 0.6 else "中" if score >= 0.35 else "弱"

    def _shooting_text(self) -> str:
        from . import tennis_review
        report = tennis_review.build(self.video_id, self.pool, float(self.session.video_duration or 0) or 1.0)
        text = f"回看候选 {report['candidate_count']} 个。"
        try:
            busy = [p for p in report["periods"] if p["count"]]
            if busy:
                text += "\n重点时段：" + "；".join(f"{p['label']} {p['count']} 个候选" for p in busy) + "。"
        except Exception:  # the summary above is enough to answer
            pass
        return text

    def _moments_text(self) -> str:
        lines = []
        for item in self.pool:
            line = (f"{item['moment_id']} {_clock(item['start_time'])}–{_clock(item['end_time'])} "
                    f"{item.get('display_name') or EVENT_TYPE_ZH.get(item['event_type'], '网球动作')} {self._strength(item)}")
            if item.get("ai"):
                review = item["ai"]
                line += f" | AI看过：{review['action']}，精彩度 {review['highlight']}/5，{review['note']}"
            if item.get("description"):
                line += f" | 画面：{item['description']}"
            lines.append(line)
        return "\n".join(lines) or "（没有识别到候选）"

    def _plan_text(self) -> str:
        plan = self.session.plan
        if not plan:
            return "还没有生成。"
        style = self.session.edit_style
        lines = [f"时长 {float(plan.get('duration', plan['target_duration'])):.1f} 秒；节奏 {PACE_ZH[style['pace']]}；"
                 f"转场 {TRANSITION_ZH[style['transition']]}；字幕{'开' if style['captions'] else '关'}；"
                 f"慢动作{'开' if style['slow_motion'] else '关'}；调色 {style.get('look', 'vivid')}；"
                 f"推镜{'开' if style.get('zoom') else '关'}；配乐 {style.get('music', 'none')}；"
                 f"标题 {style.get('title') if style.get('title') is not None else '默认'}。"]
        number = 0
        for clip in plan.get("clips", []):
            if clip.get("replay"):
                lines.append(f"  （慢动作重播 {clip.get('label', '')}）")
                continue
            number += 1
            moment = self._moment_of(clip.get("event_id")) or "?"
            lines.append(f"  第 {number} 段 = {moment}，原片 {_clock(clip['start_time'])}–{_clock(clip['end_time'])} {clip.get('label', '')}")
        s = self.settings
        known = self._by_moment()

        def describe(ids: list[str]) -> str:
            return "、".join(f"{x}（原片 {_clock(known[x]['start_time'])}）" for x in ids if x in known)
        if s["include"]:
            lines.append("  用户要求保留：" + describe(s["include"]))
        if s["exclude"]:
            lines.append("  用户已去掉：" + describe(s["exclude"]))
        if s["start_after"] is not None or s["end_before"] is not None:
            lines.append(f"  限定范围：{s['start_after'] or 0:g}–{s['end_before'] or self.session.video_duration:g} 秒")
        return "\n".join(lines)

    def _system(self) -> dict[str, str]:
        content = SYSTEM_PROMPT.format(video_name=self.video_name, duration=round(self.session.video_duration, 1),
                                       shooting=self._shooting_text(), moments=self._moments_text(), plan=self._plan_text())
        return {"role": "system", "content": content}

    # ---------- the single tool ----------
    def _validated(self, args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        """Merge model arguments over current settings; reject nothing silently."""
        settings = dict(self.settings)
        style = dict(self.session.edit_style)
        notes: list[str] = []
        known = self._by_moment()
        if "duration" in args:
            try:
                settings["duration"] = max(5.0, min(60.0, float(args["duration"])))
            except (TypeError, ValueError):
                notes.append("时长无效，保持原值")
        if args.get("focus") in {"made", "shots", "all"}:
            settings["focus"] = args["focus"]
        if "clip_count" in args:
            try:
                settings["clip_count"] = max(0, min(20, int(args["clip_count"])))
            except (TypeError, ValueError):
                notes.append("片段数无效，保持原值")
        if args.get("reset_choices") is True:
            settings.update(include=[], exclude=[], start_after=None, end_before=None)

        def moment_ids(key: str) -> list[str]:
            raw = args.get(key)
            ids = [str(x) for x in raw] if isinstance(raw, list) else []
            unknown = [x for x in ids if x not in known]
            if unknown:
                notes.append(f"忽略了不存在的时刻 {'、'.join(unknown)}")
            return [x for x in ids if x in known]

        removed = moment_ids("remove_moments")
        if isinstance(args.get("remove_segments"), list):
            current = [self._moment_of(c.get("event_id")) for c in self.session._plays(self.session.plan or {})]
            for number in args["remove_segments"]:
                try:
                    index = int(number) - 1
                except (TypeError, ValueError):
                    index = -1
                if 0 <= index < len(current) and current[index]:
                    removed.append(current[index])
                else:
                    notes.append(f"当前没有第 {number} 段")
        pinned = moment_ids("pin_moments")
        settings["include"] = [x for x in dict.fromkeys(settings["include"] + pinned) if x not in removed]
        settings["exclude"] = [x for x in dict.fromkeys(settings["exclude"] + removed) if x not in pinned]
        for key in ("start_after", "end_before"):
            if key in args:
                value = args[key]
                try:
                    settings[key] = None if value is None else max(0.0, float(value))
                except (TypeError, ValueError):
                    notes.append("时间范围无效，保持原值")
        style = normalize_edit_style({**style, **{k: args[k] for k in ("pace", "transition", "captions", "slow_motion", "look", "zoom", "music", "title", "aspect") if k in args}})
        if "music" in args and style["music"] != args["music"]:
            notes.append("没有这首音乐，已保持原声；内置可选热血/律动/轻快")
        if args.get("order") in ("time", "best_first", "climax"):
            settings["order"] = args["order"]
        caps = args.get("segment_captions")
        if isinstance(caps, dict):
            plays = self.session._plays(self.session.plan) if self.session.plan else []
            for key, text in caps.items():
                try:
                    n = int(str(key).strip("第段 "))
                except ValueError:
                    continue
                text = re.sub(r"[\x00-\x1f]", "", str(text or "")).strip()[:16]
                if 1 <= n <= len(plays) and text:
                    settings.setdefault("captions", {})[plays[n - 1].get("event_id")] = text
                else:
                    notes.append(f"没有第 {key} 段，字幕没改")
        return settings, style, notes

    def apply(self, args: dict[str, Any]) -> dict[str, Any]:
        import copy
        if self.session.plan:
            self.undo_stack = (getattr(self, "undo_stack", []) + [copy.deepcopy({
                "settings": self.settings, "plan": self.session.plan, "edit_style": self.session.edit_style,
                "clip_count": self.session.clip_count, "selected_events": self.session.selected_events,
                "order": self.session.order, "captions": self.session.captions, "commentary": self.commentary})])[-10:]
        settings, style, notes = self._validated(args)
        lo = settings["start_after"] or 0.0
        hi = settings["end_before"] or float("inf")
        if lo >= hi:
            lo, hi = 0.0, float("inf")
            settings["start_after"] = settings["end_before"] = None
            notes.append("时间范围前后颠倒，已取消范围限制")
        excluded = set(settings["exclude"])

        def eligible(item: dict[str, Any]) -> bool:
            center = (float(item["start_time"]) + float(item["end_time"])) / 2.0
            return lo <= center <= hi and item["moment_id"] not in excluded

        candidates = [r for r in (self._apply_review(m) for m in self.pool if eligible(m)) if r]
        if settings["focus"] == "shots":
            shots = [m for m in candidates if (m["ai"]["action"] in SHOT_ACTIONS if m.get("ai") else m["event_type"] in SHOT_LIKE_TYPES)]
            if shots:
                candidates = shots
            elif candidates:
                notes.append("这个范围里没有疑似投篮/扣篮，已改用所有精彩动作")
        pinned = set(settings["include"])
        chosen = {m["moment_id"]: dict(m) for m in candidates}
        for moment_id in pinned:
            chosen[moment_id] = self._apply_review(self._by_moment()[moment_id]) or dict(self._by_moment()[moment_id])
        if not chosen:
            return {"ok": False, "reason": "这个范围里没有可用的精彩时刻"}
        pool = []
        for moment_id, item in chosen.items():
            if moment_id in pinned:
                item["final_ranking_score"] = INCLUDE_BOOST + float(item.get("final_ranking_score", item.get("raw_model_score", 0.0)))
            pool.append(item)

        session = self.session
        session.edit_style = style
        session.clip_count = settings["clip_count"] or None
        session.excluded, session.time_range = set(), (None, None)   # this editor filters its own pool
        session.order = settings.get("order")
        session.captions = dict(settings.get("captions") or {})
        source = session.video_duration or max(float(x["end_time"]) for x in pool) + 2.0
        meta = duration_meta(settings["duration"], source)
        self._stage("RENDERING")
        plan = session._rebuild(pool, meta)
        for clip in plan["clips"]:
            # The boost only orders selection; keep the real model score on record.
            if float(clip.get("raw_model_score", 0.0)) >= INCLUDE_BOOST:
                clip["raw_model_score"] = round(float(clip["raw_model_score"]) - INCLUDE_BOOST, 6)
                clip["pinned"] = True
        output = session._render()
        self.settings = settings
        self.commentary = ""
        plays = session._plays(plan)
        result = {
            "ok": True,
            "duration": round(float(output["duration"]), 1),
            "clips": [{"segment": i + 1, "moment": self._moment_of(c.get("event_id")),
                       "source": f"{_clock(c['start_time'])}–{_clock(c['end_time'])}", "action": c.get("label")}
                      for i, c in enumerate(plays)],
            "slow_motion_replay": any(c.get("replay") for c in plan["clips"]),
            "style": {k: style.get(k) for k in ("pace", "transition", "captions", "look", "zoom", "music", "title", "aspect")},
        }
        if settings["clip_count"] and len(plays) < settings["clip_count"]:
            notes.append(f"可用素材只够 {len(plays)} 段")
        if meta["duration_note"]:
            notes.append(meta["duration_note"])
        if notes:
            result["notes"] = notes
        session._trace("ai_edit_highlight", {"arguments": args, "settings": settings, "result": result})
        return result
    # ---------- frame review ----------
    def inspect(self, moment_ids: list[str]) -> dict[str, Any]:
        """Show the vision model real frames of candidate moments; cache what it sees."""
        known = self._by_moment()
        wanted = [m for m in dict.fromkeys(str(x) for x in moment_ids) if m in known][:8]
        if not wanted:
            return {"ok": False, "reason": "没有可以查看的时刻"}
        fresh = [m for m in wanted if not known[m].get("ai")]
        found: dict[str, dict[str, Any]] = {}

        def review(batch: list[str]) -> dict[str, dict[str, Any]]:
            image = _contact_sheet(VIDEO_IDS[self.video_id], [(m, float(known[m]["start_time"]), float(known[m]["end_time"])) for m in batch])
            # "minimal" reasoning: with none the model stops looking and calls
            # everything filler; "low" is right but ~4x slower.
            message = self._complete([{"role": "user", "content": [
                {"type": "text", "text": INSPECT_PROMPT + "\n本图的时刻：" + "、".join(batch)},
                {"type": "image_url", "image_url": {"url": image}}]}], tools=False, max_tokens=1600, json_mode=True, effort="minimal")
            return _parse_inspection(message.content or "", set(batch))

        batches = [fresh[i:i + INSPECT_BATCH] for i in range(0, len(fresh), INSPECT_BATCH)]
        if batches:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(batches)) as pool:
                for part in pool.map(review, batches):
                    found.update(part)
        if found:
            _store_inspections(self.video_id, {_moment_key(known[m]): review for m, review in found.items()})
            for m, review in found.items():
                known[m]["ai"] = review
        self.session._trace("ai_inspect_moments", {"requested": wanted, "reviewed": found})
        return {"ok": True, "moments": [{"moment": m, **known[m]["ai"]} if known[m].get("ai") else {"moment": m, "note": "没看清"}
                                        for m in wanted]}

    @staticmethod
    def _apply_review(item: dict[str, Any]) -> dict[str, Any] | None:
        """Rank with what the AI saw: skip crowd/ads/replays, favour real highlights."""
        review = item.get("ai")
        if not review:
            return item
        if not review.get("tennis", True) or review.get("highlight", 3) <= 1:
            return None
        item = dict(item)
        item["final_ranking_score"] = 0.35 * float(item.get("raw_model_score", 0.0)) + 0.65 * (review["highlight"] / 5)
        # Detector shots already carry a plain name with the result ("跳投 · 进了");
        # only the vaguer X3D moments take the AI's "疑似…" wording.
        if item.get("evidence_source") != "LOCAL_DETECTOR":
            item["display_name"] = _review_label(review)
        return item

    def _tool(self, call: Any) -> tuple[dict[str, Any], bool]:
        """Run one model tool call; returns (result, edited)."""
        try:
            args = json.loads(call.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("bad arguments")
            if call.function.name == "edit_highlight":
                return self.apply(args), True
            if call.function.name == "inspect_moments":
                return self.inspect(list(args.get("moments") or [])), False
            if call.function.name == "get_game_stats":
                return self.game_stats(), False
            if call.function.name == "undo_edit":
                return self.undo(), True
            raise ValueError("unknown tool")
        except BudgetExceeded:
            raise
        except Exception as exc:  # render or argument failure: tell the model, then the user
            self.session._trace("ai_tool_failure", {"tool": call.function.name, "error": type(exc).__name__, "message": str(exc)})
            return {"ok": False, "reason": f"执行失败（{type(exc).__name__}）"}, False

    def _stage(self, name: str) -> None:
        hook = getattr(self.session, "stage_hook", None)
        if hook:
            hook(name)

    # ---------- model calls ----------
    def _complete(self, messages: list[dict[str, Any]], tools: bool, max_tokens: int, json_mode: bool = False,
                  effort: str = "low") -> Any:
        if len(self.usage) >= MAX_REQUESTS_PER_SESSION:
            raise BudgetExceeded("session request budget reached")
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                  "temperature": 0.2, "extra_body": {"reasoning_effort": effort}}
        if tools:
            kwargs.update(tools=[EDIT_TOOL, INSPECT_TOOL, STATS_TOOL, UNDO_TOOL], tool_choice="auto")
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        began = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        usage = response.usage.model_dump() if response.usage else {}
        with self._usage_lock:
            self.usage.append({"seconds": round(time.perf_counter() - began, 2), "tools": tools,
                               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                               "credits": usage.get("credits")})
        self._save()
        return response.choices[0].message

    def game_stats(self) -> dict[str, Any]:
        from . import tennis_review
        report = tennis_review.build(self.video_id, self.pool, float(self.session.video_duration or 0) or 1.0)
        return {"ok": True, "candidate_count": report["candidate_count"], "notes": report["notes"],
                "periods": [{k: p[k] for k in ("label", "count", "avg_score")} for p in report["periods"]],
                "moments": report["moments"][:60], "caveat": report["caveat"]}

    def undo(self) -> dict[str, Any]:
        stack = getattr(self, "undo_stack", [])
        if not stack:
            return {"ok": False, "reason": "没有可以撤销的修改"}
        snap = stack.pop()
        self.settings, self.commentary = snap["settings"], snap["commentary"]
        s = self.session
        s.plan, s.edit_style, s.clip_count, s.selected_events = snap["plan"], snap["edit_style"], snap["clip_count"], snap["selected_events"]
        s.order, s.captions = snap["order"], snap["captions"]
        s._write_plan()
        output = s._render()
        return {"ok": True, "duration": round(float(output["duration"]), 1), "segments": len(s._plays(s.plan)), "undone": True}

    def turn(self, text: str, edit_style: dict[str, Any] | None = None, selected_segment: int | None = None) -> dict[str, Any]:
        text = (text or "").strip()
        plays = len(self.session._plays(self.session.plan)) if self.session.plan else 0
        if selected_segment and 1 <= selected_segment <= plays and text:
            text = f"（我在页面上点选了第 {selected_segment} 段）{text}"
        if not text:
            return self._result("想怎么剪？比如“来个 20 秒的扣篮集锦”。", changed=False)
        if edit_style is not None:
            self.session.edit_style = normalize_edit_style(edit_style)
        edited = False
        try:
            messages = [*self.history[-HISTORY_MESSAGES:], {"role": "user", "content": text}]
            reply = ""
            for _ in range(MAX_TOOL_ROUNDS):
                message = self._complete([self._system(), *messages], tools=True, max_tokens=600)
                calls = list(message.tool_calls or [])
                if not calls:
                    reply = (message.content or "").strip()
                    break
                assistant = {"role": "assistant", "content": message.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}} for c in calls]}
                reasoning = getattr(message, "reasoning_content", None)
                if reasoning:
                    assistant["reasoning_content"] = reasoning
                messages.append(assistant)
                last_result: dict[str, Any] = {}
                for index, call in enumerate(calls):
                    if index >= 3:
                        result = {"ok": False, "reason": "一次最多执行 3 个操作"}
                    else:
                        result, did_edit = self._tool(call)
                        edited = edited or (did_edit and result.get("ok", False))
                    last_result = result
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
            else:
                try:
                    reply = (self._complete([self._system(), *messages], tools=False, max_tokens=300).content or "").strip()
                except BudgetExceeded:
                    raise
                except Exception:
                    reply = self._template(last_result) if last_result.get("clips") or not last_result.get("ok", True) else ""
            reply = reply or "好的。"
        except BudgetExceeded:
            return self._result("这次对话已经很长了，点“新对话”重新开始吧。", changed=False)
        except Exception as exc:
            return self._fallback(text, exc)
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return self._result(reply, changed=edited)

    def generate(self, duration: float, focus: str, edit_style: dict[str, Any] | None = None) -> dict[str, Any]:
        """The page's "生成" button: settings come from the form, the model writes commentary."""
        self.settings.update(include=[], exclude=[], clip_count=0, start_after=None, end_before=None)
        args = {"duration": duration, "focus": focus if focus == "shots" else "all", **normalize_edit_style(edit_style)}
        self._stage("AI_LOOKING")
        shortlist = sorted((m for m in self.pool if not m.get("ai") and (focus == "all" or m["event_type"] in SHOT_LIKE_TYPES)),
                           key=lambda m: float(m.get("raw_model_score", 0.0)), reverse=True)[:8]
        if shortlist:
            try:
                self.inspect([m["moment_id"] for m in shortlist])
            except BudgetExceeded:
                raise
            except Exception as exc:  # seeing frames is a bonus; the edit still works without it
                self.session._trace("ai_inspect_failure", {"error": type(exc).__name__})
        result = self.apply(args)
        if not result.get("ok"):
            raise ValueError(result.get("reason", "没有可用的精彩时刻"))
        self._stage("AI_WRITING")
        try:
            clips = "\n".join(f"第 {c['segment']} 段：原片 {c['source']} {c['action']}"
                              + (f"（AI看到：{self._by_moment()[c['moment']]['ai']['note']}）"
                                 if c.get("moment") and self._by_moment()[c["moment"]].get("ai") else "")
                              + (f"（画面：{self._by_moment()[c['moment']].get('description')}）"
                                 if c.get("moment") and self._by_moment()[c["moment"]].get("description") else "")
                              for c in result["clips"])
            message = self._complete([{"role": "system", "content": COMMENTARY_PROMPT},
                                      {"role": "user", "content": f"{self.video_name}，{result['duration']} 秒集锦：\n{clips}"}],
                                     tools=False, max_tokens=260)
            self.commentary = (message.content or "").strip()
        except Exception:
            self.commentary = ""
        reply = self.commentary or self._template(result)
        self.history += [{"role": "user", "content": f"生成 {duration:g} 秒{ {'shots': '发球/击球'}.get(focus, '精彩') }集锦"},
                         {"role": "assistant", "content": reply}]
        return self._result(reply, changed=True)

    # ---------- helpers ----------
    def _fallback(self, text: str, exc: Exception) -> dict[str, Any]:
        """Model unreachable: the basic editor still does the job."""
        self.session._trace("ai_unavailable", {"error": type(exc).__name__})
        before = (self.session.plan or {}).get("duration")
        local = self.session.ask(text)
        plan = self.session.plan or {}
        if plan:
            self.settings["duration"] = float(plan.get("target_duration", self.settings["duration"]))
        reply = "AI 剪辑师暂时连不上，先用基础剪辑帮你处理：" + local.get("reply", "")
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return self._result(reply, changed="output" in local or before != plan.get("duration"))

    @staticmethod
    def _template(result: dict[str, Any]) -> str:
        if not result.get("ok"):
            return f"这次没剪成：{result.get('reason', '请换个说法再试')}。"
        note = ("（" + "；".join(result["notes"]) + "）") if result.get("notes") else ""
        return f"剪好了：{result['duration']} 秒，共 {len(result['clips'])} 段。{note}"

    def _result(self, reply: str, changed: bool) -> dict[str, Any]:
        self._save()
        plan = self.session.plan
        output = None
        if changed and plan and plan.get("segments"):
            output = {"path": str((self.session.output_dir / "highlights.mp4").relative_to(ROOT)),
                      "duration": plan.get("duration"), "segments": plan["segments"], "edit_style": plan.get("edit_style"),
                      "duration_note": plan.get("duration_note", ""), "commentary": self.commentary}
        from .edit_ops import suggestions
        return {"reply": reply, "session_id": self.session_id, "mode": "ai", "edit_style": dict(self.session.edit_style),
                "output": output, "plan": plan, "commentary": self.commentary,
                "suggestions": suggestions(plan, any(m.get("outcome") == "made" for m in self.pool), ai=True,
                                           can_undo=bool(getattr(self, "undo_stack", []))),
                "can_undo": bool(getattr(self, "undo_stack", []))}

    def _save(self) -> None:
        directory = self.session.output_dir
        (directory / "ai_history.json").write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")
        (directory / "api_usage.json").write_text(json.dumps(self.usage, ensure_ascii=False, indent=2), encoding="utf-8")
        (directory / "tool_trace.json").write_text(json.dumps(self.session.traces, ensure_ascii=False, indent=2), encoding="utf-8")
