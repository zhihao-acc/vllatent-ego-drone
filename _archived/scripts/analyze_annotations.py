#!/usr/bin/env python3
"""Compute comprehensive statistics across AerialVLN annotation splits."""

import json
import sys
from collections import Counter

DATA_DIR = "/home/zh/CODE/vllatent-ego-drone/data/aerialvln_json"
SPLITS = ["train", "val_seen", "val_unseen", "test"]
ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
    4: "MOVE_UP",
    5: "MOVE_DOWN",
    6: "TURN_LEFT_SMALL",
    7: "TURN_RIGHT_SMALL",
}


def get_instruction_text(ep):
    instr = ep.get("instruction") or ep.get("instructions")
    if isinstance(instr, dict):
        return instr.get("instruction_text", "")
    if isinstance(instr, str):
        return instr
    if isinstance(instr, list):
        return " ".join(str(i) for i in instr)
    return ""


def get_trajectory_id(ep):
    return ep.get("trajectory_id") or ep.get("path_id") or ep.get("episode_id")


def analyze_split(name, episodes):
    print()
    print("=" * 70)
    print(f"  SPLIT: {name}  ({len(episodes)} episodes)")
    print("=" * 70)

    # 1. Total episodes
    print(f"\n1. Total episodes: {len(episodes)}")

    # 2. Unique trajectory_ids
    traj_ids = set(get_trajectory_id(ep) for ep in episodes)
    print(f"2. Unique trajectory_ids: {len(traj_ids)}")

    # 3. Unique scene_ids
    scene_ids = sorted(set(ep["scene_id"] for ep in episodes))
    print(f"3. Unique scene_ids ({len(scene_ids)}): {scene_ids}")

    # 4-5. Instruction word count stats
    word_counts = [len(get_instruction_text(ep).split()) for ep in episodes]
    avg_wc = sum(word_counts) / len(word_counts) if word_counts else 0
    print(f"4. Avg instruction word count: {avg_wc:.1f}")
    print(f"5. Min/Max instruction word count: {min(word_counts)} / {max(word_counts)}")

    # 6-7. Reference path length
    path_lens = [len(ep.get("reference_path", [])) for ep in episodes]
    avg_pl = sum(path_lens) / len(path_lens) if path_lens else 0
    print(f"6. Avg reference_path length: {avg_pl:.1f} steps")
    print(f"7. Min/Max reference_path length: {min(path_lens)} / {max(path_lens)}")

    # 8. Action type counts
    action_counter = Counter()
    total_actions = 0
    for ep in episodes:
        for a in ep.get("actions", []):
            action_counter[a] += 1
            total_actions += 1
    print(f"8. Action distribution (total={total_actions}):")
    for a in range(8):
        cnt = action_counter.get(a, 0)
        pct = (cnt / total_actions * 100) if total_actions > 0 else 0
        label = ACTION_NAMES.get(a, f"ACTION_{a}")
        print(f"   {a} ({label:>20s}): {cnt:>8d}  ({pct:5.2f}%)")

    # 9. Instructions per trajectory
    ipt = len(episodes) / len(traj_ids) if traj_ids else 0
    print(f"9. Instructions per trajectory: {ipt:.2f}")

    # 10. Verify reference_path element width
    widths = Counter()
    for ep in episodes:
        for pose in ep.get("reference_path", []):
            widths[len(pose)] += 1
    all6 = set(widths.keys()) == {6}
    print(f"10. ref_path element widths: {dict(widths)}  [{'OK' if all6 else 'WARNING'}]")

    # 11. Verify start_rotation width
    rot_widths = Counter()
    for ep in episodes:
        rot_widths[len(ep.get("start_rotation", []))] += 1
    all4 = set(rot_widths.keys()) == {4}
    print(f"11. start_rotation widths: {dict(rot_widths)}  [{'OK' if all4 else 'WARNING'}]")

    # 12. Sample reference_path pitch/roll
    print("12. Sample ref_path (first 3 eps, first 2 poses):")
    for i, ep in enumerate(episodes[:3]):
        eid = ep.get("episode_id", "?")
        rp = ep.get("reference_path", [])
        print(f"   Ep {eid}:")
        for j, pose in enumerate(rp[:2]):
            x, y, z, pitch, roll, yaw = pose
            print(
                f"     [{j}] x={x:.2f} y={y:.2f} z={z:.2f} "
                f"pitch={pitch} roll={roll} yaw={yaw:.6f}"
            )
        pitches = [p[3] for p in rp]
        rolls = [p[4] for p in rp]
        nzp = sum(1 for p in pitches if abs(p) > 1e-6)
        nzr = sum(1 for r in rolls if abs(r) > 1e-6)
        print(f"     non-zero pitch: {nzp}/{len(pitches)}, non-zero roll: {nzr}/{len(rolls)}")

    return {
        "n": len(episodes),
        "traj": traj_ids,
        "scenes": set(scene_ids),
        "ac": action_counter,
        "ta": total_actions,
    }


def main():
    grand_traj = set()
    grand_scenes = set()
    grand_ac = Counter()
    grand_ta = 0
    grand_n = 0
    split_sc = {}

    for split in SPLITS:
        path = f"{DATA_DIR}/{split}.json"
        print(f"Loading {split}.json ...", end=" ", flush=True)
        with open(path) as f:
            raw = json.load(f)
        eps = raw["episodes"] if isinstance(raw, dict) and "episodes" in raw else raw
        print(f"{len(eps)} episodes loaded.")

        st = analyze_split(split, eps)
        grand_traj |= st["traj"]
        grand_scenes |= st["scenes"]
        grand_ac += st["ac"]
        grand_ta += st["ta"]
        grand_n += st["n"]
        split_sc[split] = st["scenes"]

    # ---- Grand Totals ----
    print()
    print("#" * 70)
    print("  GRAND TOTALS")
    print("#" * 70)
    print(f"\nTotal episodes: {grand_n}")
    print(f"Total unique trajectories (union): {len(grand_traj)}")
    print(f"Total unique scenes (union): {len(grand_scenes)}")
    print(f"All scenes sorted: {sorted(grand_scenes)}")

    print(f"\nGrand action distribution (total={grand_ta}):")
    for a in range(8):
        cnt = grand_ac.get(a, 0)
        pct = (cnt / grand_ta * 100) if grand_ta > 0 else 0
        label = ACTION_NAMES.get(a, f"ACTION_{a}")
        print(f"  {a} ({label:>20s}): {cnt:>10d}  ({pct:5.2f}%)")

    print("\nScene overlap between splits:")
    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i + 1 :]:
            ov = split_sc[s1] & split_sc[s2]
            print(f"  {s1} & {s2}: {sorted(ov) if ov else '(none)'}")

    print("\nScenes unique to each split:")
    for s in SPLITS:
        others = set()
        for s2 in SPLITS:
            if s2 != s:
                others |= split_sc[s2]
        uniq = split_sc[s] - others
        print(f"  {s}: {sorted(uniq) if uniq else '(none)'}")

    # ---- Audit Report ----
    print()
    print("#" * 70)
    print("  AUDIT REPORT SUMMARY")
    print("#" * 70)

    with open(f"{DATA_DIR}/audit_report.json") as f:
        audit = json.load(f)

    print(f"\nTotal audit entries: {len(audit)}")
    ok_c = sum(1 for a in audit if a.get("ok"))
    print(f"OK: {ok_c}, NOT OK: {len(audit) - ok_c}")
    print(
        f"alignment_ok: "
        f"{sum(1 for a in audit if a.get('alignment_ok'))}/{len(audit)}"
    )
    print(
        f"tuple_complete: "
        f"{sum(1 for a in audit if a.get('tuple_complete'))}/{len(audit)}"
    )
    print(
        f"all_action_classes_present: "
        f"{sum(1 for a in audit if a.get('all_action_classes_present'))}/{len(audit)}"
    )

    qo = Counter()
    rpo = Counter()
    for a in audit:
        q = a.get("quaternion", {})
        qo[q.get("start_rotation_order", "?")] += 1
        rpo[q.get("reference_path_order", "?")] += 1
    print(f"start_rotation_order: {dict(qo)}")
    print(f"reference_path_order: {dict(rpo)}")

    asc = Counter(a.get("scene_id") for a in audit)
    print(f"Audit scenes: {dict(sorted(asc.items()))}")

    sp_c = Counter()
    for a in audit:
        for sp in a.get("splits_present", []):
            sp_c[sp] += 1
    print(f"splits_present: {dict(sp_c)}")

    hdm = sum(1 for a in audit if a.get("delta_mismatches"))
    print(f"delta_mismatches present: {hdm}/{len(audit)}")

    lic = Counter(a.get("license", "?") for a in audit)
    print(f"Licenses: {dict(lic)}")

    sr_set = set()
    for a in audit:
        sr = a.get("scene_id_range")
        if sr:
            sr_set.add(tuple(sr) if isinstance(sr, list) else sr)
    print(f"scene_id_range: {sr_set}")

    if len(audit) - ok_c > 0:
        print("\nSample NOT-OK entry:")
        for a in audit:
            if not a.get("ok"):
                print(f"  {json.dumps(a)[:500]}")
                break

    print()
    print("=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
