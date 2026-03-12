"""
GoalCut 因果推理链验证器（算法 E，O-23）

核心思想：篮球进球不是孤立事件，而是一个时间因果链。
  - 真实进球前必有「前因」（投篮动作/球体运动）
  - 真实进球后必有「后果」（庆祝反应/音频事件/篮网形变）

通过检查候选事件前后的因果信号是否存在，区分真正进球与孤立噪声。

三种得分方式的因果链模型：
  1. 投篮（2分/3分/罚球投篮）：球体抛物线飞行 → 穿筐 → 入网声/欢呼
  2. 上篮：人体快速接近篮筐 → 近距投入/扣入 → 欢呼/庆祝
  3. 罚球：场面静止（无对抗）→ 球穿筐 → 温和反应

集成位置：跨通道互证之后、merge_events_with_cooldown 之前。
"""

from typing import Dict, List, Tuple, Optional


# ============================================================
# 前因信号定义（Precursor Signals）
# 检测进球前的运动/动作信号
# ============================================================

PRECURSOR_SIGNALS = {
    # 球体运动轨迹（投篮时球在空中飞行）
    "ball_motion": {
        "window": (-3.0, -0.3),     # 进球前 0.3~3s
        "sources": {"几何判定", "辅助判定", "启发式"},
        "weight": 1.0,
    },
    # 人体接近篮筐（上篮/扣篮动作）
    "person_motion": {
        "window": (-4.0, -0.3),     # 进球前 0.3~4s
        "sources": {"人体运动"},
        "weight": 0.8,
    },
    # 篮筐区域运动能量爆发
    "motion_burst": {
        "window": (-2.0, 0.5),      # 进球前后 2s~0.5s
        "sources": {"运动突变"},
        "weight": 0.5,              # 权重低：运动突变本身噪声大
    },
}

# ============================================================
# 后果信号定义（Consequence Signals）
# 检测进球后的反应信号
# ============================================================

CONSEQUENCE_SIGNALS = {
    # 进球后的音频反应
    "audio_reaction": {
        "window": (0.0, 4.0),       # 进球后 0~4s
        "sources": {"入网声", "叫好声", "击掌声", "哨声"},
        "weight": 1.0,
    },
    # 球员庆祝行为
    "celebration": {
        "window": (0.5, 5.0),       # 进球后 0.5~5s
        "sources": {"庆祝动作"},
        "weight": 0.8,
    },
    # 篮网形变（球穿网的直接物理证据）
    "net_deform": {
        "window": (-0.5, 1.5),      # 进球前后 0.5s~1.5s（几乎同步）
        "sources": {"篹网有向穿越", "篮网有向穿越", "篹网形变", "篮网形变"},
        "weight": 1.0,
    },
}


def _extract_channel_name(detail: str) -> str:
    """从事件 detail 字符串中提取通道名称。

    例如:
        "几何判定: ball穿越hoop" → "几何判定"
        "人体运动(left)" → "人体运动"
        "入网声: 2.5kHz peak" → "入网声"
        "庆祝动作: 急停=0.35, 聚集=0.20" → "庆祝动作"
    """
    src = detail.split(":")[0].strip()
    paren_idx = src.find("(")
    if paren_idx > 0:
        src = src[:paren_idx].strip()
    return src


def _count_active_channels(profile) -> int:
    """统计当前 AlgorithmProfile 中启用的通道数。"""
    if profile is None:
        return 7  # 默认假设全通道
    count = 0
    for ch_id in ("1a", "1b", "2", "3", "4", "5", "6", "7"):
        ch_cfg = profile.channels.get(ch_id, {})
        if isinstance(ch_cfg, dict) and ch_cfg.get("enabled", True):
            count += 1
        elif isinstance(ch_cfg, bool) and ch_cfg:
            count += 1
    return max(count, 1)


def evaluate_causal_chain(
    candidate_event: Dict,
    all_events: List[Dict],
    video_duration: float,
    profile=None,
) -> Tuple[float, str, str]:
    """评估单个候选进球事件的因果链完整性。

    Args:
        candidate_event: 待评估的候选进球事件
            必须包含 "timestamp" 和 "detail" 字段
        all_events: 所有通道产生的候选事件列表（含 candidate_event 自身）
        video_duration: 视频总时长（秒），用于边界条件处理
        profile: AlgorithmProfile 对象（可选），用于判断通道数

    Returns:
        (causal_score, goal_type, reason)
        - causal_score: 因果链调整分数 [-0.15, +0.12]
        - goal_type: 推测的进球类型
            "shot"        投篮（2分/3分）
            "layup"       上篮/扣篮
            "free_throw"  罚球
            "unknown"     无法判断
        - reason: 可读的评分原因描述
    """
    t = candidate_event["timestamp"]
    self_detail = _extract_channel_name(candidate_event.get("detail", ""))

    # ---- 搜集前因信号 ----
    precursor_hits = {}  # sig_name -> (confidence, detail)

    for sig_name, sig_cfg in PRECURSOR_SIGNALS.items():
        w_start, w_end = sig_cfg["window"]

        # 边界条件：视频开头，前因窗口被截断
        effective_start = max(0, t + w_start)
        effective_end = t + w_end
        if effective_start >= effective_end:
            continue

        for other in all_events:
            if other is candidate_event:
                continue
            ot = other["timestamp"]
            if effective_start <= ot <= effective_end:
                other_channel = _extract_channel_name(other.get("detail", ""))
                if other_channel in sig_cfg["sources"]:
                    # 取最高置信度的匹配
                    existing = precursor_hits.get(sig_name)
                    if existing is None or other["confidence"] > existing[0]:
                        precursor_hits[sig_name] = (
                            other["confidence"], other_channel
                        )
                    break  # 每种前因信号只需找到一个即可

    # ---- 搜集后果信号 ----
    consequence_hits = {}  # sig_name -> (confidence, detail)

    for sig_name, sig_cfg in CONSEQUENCE_SIGNALS.items():
        w_start, w_end = sig_cfg["window"]

        # 边界条件：视频结尾，后果窗口被截断
        effective_start = t + w_start
        effective_end = min(video_duration, t + w_end)
        if effective_start >= effective_end:
            continue

        for other in all_events:
            if other is candidate_event:
                continue
            ot = other["timestamp"]
            if effective_start <= ot <= effective_end:
                other_channel = _extract_channel_name(other.get("detail", ""))
                if other_channel in sig_cfg["sources"]:
                    existing = consequence_hits.get(sig_name)
                    if existing is None or other["confidence"] > existing[0]:
                        consequence_hits[sig_name] = (
                            other["confidence"], other_channel
                        )
                    break

    # ---- 汇总信号 ----
    precursor_count = len(precursor_hits)
    consequence_count = len(consequence_hits)
    total_signals = precursor_count + consequence_count

    has_ball_motion = "ball_motion" in precursor_hits
    has_person_motion = "person_motion" in precursor_hits
    has_audio = "audio_reaction" in consequence_hits
    has_celebration = "celebration" in consequence_hits
    has_net = "net_deform" in consequence_hits

    # ---- 推测进球类型 ----
    if has_person_motion and not has_ball_motion:
        goal_type = "layup"
    elif has_ball_motion and not has_person_motion:
        goal_type = "shot"
    elif precursor_count == 0 and consequence_count >= 1:
        # 前因安静但后果存在 → 可能罚球（或投篮前因被遮挡未检出）
        goal_type = "free_throw"
    elif has_ball_motion and has_person_motion:
        goal_type = "layup"  # 同时有球和人运动，更像上篮
    else:
        goal_type = "unknown"

    # ---- 边界条件：视频首尾宽容处理 ----
    at_video_start = t < 3.0
    at_video_end = t > video_duration - 3.0

    # 活跃通道数影响惩罚力度
    active_channels = _count_active_channels(profile)
    is_minimal_mode = active_channels <= 3

    # ---- 因果评分 ----
    reason_parts = []

    # 检查后果信号是否只有击掌声（野球场击掌声密度极高，不可靠）
    consequence_only_clap = (
        consequence_count >= 1 and
        all(detail[1] == "击掌声" for detail in consequence_hits.values())
    )

    if total_signals >= 3:
        if consequence_only_clap and precursor_count < 2:
            # 后果只有击掌声且前因不够强 → 降级
            causal_score = +0.03
            reason_parts.append(f"降级因果链(后果仅击掌声,"
                                f"{precursor_count}前因+{consequence_count}后果)")
        else:
            causal_score = +0.12
            reason_parts.append(f"强因果链({precursor_count}前因+{consequence_count}后果)")
    elif total_signals == 2:
        if consequence_only_clap:
            # 后果只有击掌声 → 降级为弱因果链（中性，不惩罚也不boost）
            causal_score = 0.0
            reason_parts.append(f"降级弱因果链(后果仅击掌声,"
                                f"{precursor_count}前因+{consequence_count}后果,中性)")
        else:
            causal_score = +0.08
            reason_parts.append(f"中等因果链({precursor_count}前因+{consequence_count}后果)")
    elif total_signals == 1:
        causal_score = 0.0
        reason_parts.append(f"弱因果链({precursor_count}前因+{consequence_count}后果,中性)")
    else:
        # 完全孤立事件
        if at_video_start or at_video_end:
            # 视频首尾：窗口被截断，不做全额惩罚
            causal_score = -0.05
            reason_parts.append("孤立事件(视频边界，轻惩罚)")
        elif is_minimal_mode:
            # minimal 模式通道少，信号来源有限，惩罚减半
            causal_score = -0.10
            reason_parts.append("孤立事件(通道少，惩罚减半)")
        else:
            causal_score = -0.20
            reason_parts.append("孤立噪声(前后均无因果信号)")

    # 构建详细原因描述
    if precursor_hits:
        pre_desc = ", ".join(
            f"{name}({detail[1]})" for name, detail in precursor_hits.items()
        )
        reason_parts.append(f"前因=[{pre_desc}]")
    if consequence_hits:
        con_desc = ", ".join(
            f"{name}({detail[1]})" for name, detail in consequence_hits.items()
        )
        reason_parts.append(f"后果=[{con_desc}]")

    reason = "; ".join(reason_parts)

    return causal_score, goal_type, reason


def apply_causal_validation(
    all_candidate_events: List[Dict],
    video_duration: float,
    profile=None,
) -> List[Dict]:
    """对所有候选事件应用因果推理链验证。

    在跨通道互证之后、事件合并之前调用。
    修改 all_candidate_events 中每个事件的 confidence，
    并添加 causal_score / goal_type 字段。

    Args:
        all_candidate_events: 经过跨通道互证后的候选事件列表（会被原地修改）
        video_duration: 视频总时长
        profile: AlgorithmProfile 对象

    Returns:
        修改后的候选事件列表（同一对象）
    """
    if not all_candidate_events:
        return all_candidate_events

    print(f"[causal] 因果推理链验证: {len(all_candidate_events)} 个候选事件", flush=True)

    for event in all_candidate_events:
        causal_score, goal_type, reason = evaluate_causal_chain(
            event, all_candidate_events, video_duration, profile
        )

        old_conf = event["confidence"]
        # 应用因果调整，确保置信度在 [0, 1] 范围内
        new_conf = max(0.0, min(1.0, old_conf + causal_score))
        event["confidence"] = round(new_conf, 3)

        # 附加因果链信息到事件
        event["causal_score"] = causal_score
        event["goal_type"] = goal_type

        # 日志
        if abs(causal_score) > 0.001:
            direction = "↑" if causal_score > 0 else "↓"
            print(
                f"[causal]   {event['timestamp']:.1f}s: "
                f"conf {old_conf:.3f} → {new_conf:.3f} "
                f"({direction}{abs(causal_score):.2f}), "
                f"type={goal_type}, {reason}",
                flush=True
            )
        else:
            print(
                f"[causal]   {event['timestamp']:.1f}s: "
                f"conf {old_conf:.3f} (不变), "
                f"type={goal_type}, {reason}",
                flush=True
            )

    # 统计因果验证效果
    boosted = sum(1 for e in all_candidate_events if e.get("causal_score", 0) > 0)
    penalized = sum(1 for e in all_candidate_events if e.get("causal_score", 0) < 0)
    unchanged = len(all_candidate_events) - boosted - penalized
    print(
        f"[causal] 因果验证完成: "
        f"{boosted} 提升 / {penalized} 惩罚 / {unchanged} 不变",
        flush=True
    )

    return all_candidate_events
