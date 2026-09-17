"""Export a readable figure index and methods for the return/sleep probe."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from analysis.grid_occupancy_utils import format_clock_time


def write_report(root: Path, figures: list[Path], manifest: dict, tracks: pd.DataFrame,
                 returns: pd.DataFrame, return_effects: pd.DataFrame, contacts: pd.DataFrame,
                 triggers: pd.DataFrame, diagnostics: pd.DataFrame, outcomes: pd.DataFrame,
                 wake_effects: pd.DataFrame, *, start_clock_seconds: float,
                 sensitivity_counts: pd.DataFrame | None = None,
                 sensitivity_effects: pd.DataFrame | None = None) -> None:
    fps = manifest["settings"]["fps"]
    settings = manifest["settings"]
    undirected = manifest.get("interaction_parameters", {}).get("directed") is False
    sleep_history = manifest["sleep_classifier_parameters"]["window_seconds"]
    duration = (tracks.frame_max.max() + 1) / fps
    counts = []
    for side in ("left", "right"):
        side_returns = returns[returns.side == side]
        side_triggers = triggers[(triggers.side == side) & (triggers.condition == "Recent return contact")]
        side_contacts = contacts[(contacts.side == side) & (contacts.condition == "Recent return contact")]
        counts.append({"colony": side, "clustered_ants": int((tracks.side == side).sum()),
                       "returns": len(side_returns), "resource_visit_returns": int(side_returns.resource_visit.sum()),
                       "sleeping_recipient_contacts": len(side_contacts), "matched_pairs": len(side_triggers),
                       "matched_recipient_ants": side_triggers.track_id.nunique()})
    counts = pd.DataFrame(counts)
    counts.to_csv(root / "analysis_counts.csv", index=False)
    methods = [
        f"Recording: {format_clock_time(start_clock_seconds)} to {format_clock_time(start_clock_seconds + duration)} "
        f"({duration / 3600:.2f} hours, crossing midnight). This is the recorded interval, not a complete 24-hour cycle.",
        "Cluster sleep curves use the mean of each ant's fraction of classified frames asleep. Each contributing ant has equal weight. "
        "Unknown labels never count as wake. Ant-time bins below the configured classified coverage threshold are omitted; coverage is plotted.",
        f"Sleep labels are the cached body/antenna motion classifier, with parameters recorded in settings.json. The {sleep_history:g}-second trailing history "
        "can delay wake labels relative to initial movement. Sleep here is an operational motion-based classification.",
        "Return events use the current panorama_regions.csv in tracking coordinates. The legacy colony_presence_vectors have incompatible "
        f"colony boxes for this recording and are not used. Positions require at least {100*settings['min_position_fraction']:g}% of frames "
        f"in a {settings['position_bin_seconds']:g}-second bin. Unknown position runs break a trip; they are not bridged. "
        f"A qualifying excursion has at least {settings['min_outside_seconds']:g} seconds outside and at least "
        f"{settings['min_colony_anchor_seconds']:g} seconds inside on both ends. Brief known border flicker "
        f"(<{settings['border_flicker_seconds']:g} seconds) between equal states is removed for qualification only. "
        "Time zero is then refined to the first raw outside-to-inside tracking-anchor crossing of the confirmed return residence, "
        "including brief boundary recrossings hidden by majority binning. Earlier touches followed by a long outside interval are "
        "separate visits, not the start of this residence. "
        f"Crossings across more than {settings['crossing_max_gap_frames']} missing-frame steps are excluded. Exact return, last-outside and old bin frames are exported. "
        "This is the anchor entering the annotated rectangle, not the first antenna crossing or entry into one camera's field of view.",
        f"All excursion types are pooled. Food/water-visit metadata mark at least {settings['min_resource_seconds']:g} seconds of detections in annotated resource regions; "
        "neither location nor a resource visit proves that feeding or transport occurred.",
        "Returns are followed for up to 10 minutes, with post-return curves ending at the next exit or loss of position coverage. The early-minus-late "
        "comparison pairs 0-60 seconds and 300-600 seconds from the same return and needs >=50% observed bins in both windows. Thus it describes "
        "returns with sufficiently long observed residence. Per-time contributing-ant counts are exported in the summary table. "
        "Figure 4 contains only body motion, antenna motion and new-contact rate. Figure 5's lower row counts return episodes still followed "
        "without sustained sleep, not unique ants and not a fraction awake in the whole colony; sleep, exit, lost labels or follow-up end remove an episode.",
        ("Interactions use the minimum distance between all finished skeleton segments and observed nodes; segment crossings have distance zero. "
         "The cutoff alone defines a frame-level hit, without an antenna restriction, center-radius cutoff, duplicate-pose filter or temporal gate. "
         "Each pair is stored once per frame, without a direction. " if undirected else
         "Legacy interactions test antenna nodes against all partner nodes, including antennae. ") +
        "These geometric detections do not establish physical contact. "
        f"The selected proximity cutoff is {manifest.get('interaction_distance_mm', 'unrecorded')} mm; new caches include parameter metadata. "
        "Reciprocal detections are merged into unordered pair bouts, including across contiguous "
        f"chunk boundaries. A new pair onset requires more than {settings['contact_gap_seconds']:g} seconds since the pair's previous detection. Contacts beginning at the start of "
        "an observed segment are left-censored and do not count as known onsets. Continuous neighboring/sleeping pairs contribute one bout, not repeated events. "
        f"A retained bout needs {settings['contact_min_detection_frames']} distinct detected frames; contact_audit.csv reports isolated hits. "
        "Long detection dropouts can still split a true continuous interaction. Review distance, geometry and temporal persistence in the contact viewer before interpreting waking effects.",
        f"A recipient must have sleep labels on every frame in the preceding {settings['prior_sleep_seconds']:g} seconds, both ants must be inside in the previous position bin, and the "
        f"recipient must have no contact over the preceding {settings['prior_contact_free_seconds']:g} seconds (rounded up to complete position bins). "
        f"The partner must have returned within the previous {settings['recent_return_seconds']:g} seconds "
        "and still be in the same observed residence. " +
        ("Skeleton distances do not identify an initiator or anatomical contact type; recipient_body_contact is unknown for these inputs."
         if undirected else "Legacy contact directions are pooled; recipient_body_contact flags the returner's antenna near any recipient node."),
        "The first eligible returning contact per recipient sleep bout is offered for matching. No-contact times come from the same ant, within "
        f"{settings['match_clock_seconds']/60:g} minutes, within {settings['match_distance_mm']:g} mm, local neighbor count within "
        f"{settings['match_density_difference']:g} ({settings['density_radius_mm']:g}-mm radius), preceding-bin body motion within "
        f"{settings['match_body_speed_mm_s']:g} mm/s and antenna motion within {settings['match_antenna_speed_mm_s']:g} mm/s. "
        f"Sleep age (capped at 300 seconds) is within a factor of {settings['match_sleep_age_ratio']:g}. Controls also require prior sleep and contact-free history. "
        f"Event/control times are separated by >={settings['min_control_separation_seconds']:g} seconds; control times are not reused nearby. These limits are editable in SETTINGS. "
        "Unmatched events are reported, not silently replaced by controls from different ants.",
        "A second comparator uses eligible contacts with ants without a qualifying recent return under the same matching calipers. Its matched subset "
        "can differ from the no-contact comparison. Future sleep/wake outcomes are not used to choose any control.",
        f"Recipient state/motion curves display the full observed response, including subsequent contacts. The time-to-wake analysis requires {settings['wake_sustain_seconds']:g} seconds "
        "of continuous wake labels and censors at the next new recipient contact, any unknown sleep label, colony exit/loss of position, or the end of "
        "interaction coverage. Kaplan-Meier curves use the observed risk set and assume non-informative censoring; no independent-event confidence band is shown. "
        "Probability curves stop when fewer than five events remain at risk.",
        "Paired wake-risk differences use only pairs with known outcomes at each horizon. For these and activity curves, events are averaged within ant "
        "before averaging ants, and 95% bootstrap intervals resample ants. Intervals are omitted for fewer than five contributing ants. "
        "Intervals are descriptive within each colony; two colonies do not establish "
        "a population-wide effect. Censoring and complete-pair selection can change the analyzed population at later times.",
        "Evidence for the proposed sequence requires elevated early post-return motion/contact rates and excess recipient waking over matched controls. "
        "Tracking associations alone do not establish that the contact caused waking. Compare return contacts with other contacts, inspect matching "
        "balance, censoring and video-validated contact definitions before attributing an effect to returning ants.",
    ]
    sections = [("Data counts", counts), ("Early minus late return activity", return_effects),
                ("Paired wake-risk differences", wake_effects)]
    if sensitivity_counts is not None and not sensitivity_counts.empty:
        sections.append(("Sensitivity sample sizes", sensitivity_counts))
    if sensitivity_effects is not None and not sensitivity_effects.empty:
        sections.append(("Sensitivity wake-risk differences", sensitivity_effects))
    if not outcomes.empty:
        censoring = outcomes.groupby(["side", "condition", "censor_reason"]).size().rename("n_events").reset_index()
        sections.append(("Wake follow-up and censoring", censoring))
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>0723 return and sleep analysis</title>",
            "<style>body{font:15px system-ui,sans-serif;max-width:1250px;margin:32px auto;padding:0 20px;color:#202124}"
            "img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:13px}th,td{padding:6px 9px;border:1px solid #ddd}"
            "section{overflow-x:auto;margin:28px 0}p{line-height:1.5}a{color:#126884}</style></head><body>",
            "<h1>0723 block02: colony returns and sleep</h1>"]
    for title, table in sections:
        html.append(f"<section><h2>{escape(title)}</h2>{table.to_html(index=False, float_format=lambda x: f'{x:.4g}')}</section>")
    for path in figures:
        relative = path.relative_to(root)
        pdf = relative.with_suffix(".pdf")
        html.append(f"<section><h2>{escape(path.stem.replace('_', ' '))}</h2><a href='{pdf}'>PDF</a> "
                    f"<a href='{relative}'>PNG</a><img src='{relative}' alt='{escape(path.stem)}'></section>")
    html.append("<h2>Definitions and interpretation</h2>")
    html.extend(f"<p>{escape(paragraph)}</p>" for paragraph in methods)
    html.append("</body></html>")
    (root / "index.html").write_text("\n".join(html))
    text = ["# 0723 block02 return and sleep analysis", *methods]
    for title, table in sections:
        text.extend([f"## {title}", "```text\n" + table.to_string(index=False) + "\n```"])
    (root / "report.md").write_text("\n\n".join(text) + "\n")
