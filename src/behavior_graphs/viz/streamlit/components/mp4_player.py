"""Rollout video players: one episode, one or more camera views, in sync.

A multi-view player puts every view of the episode in the same iframe and keeps
them frame-aligned in JavaScript, so the panel's timeline, cluster strip and
behavior-segment overlay describe all of them at once. Which views are *shown*
is a CSS decision, driven live by
:mod:`behavior_graphs.viz.streamlit.components.camera_views` — hidden views stay
loaded and playing rather than being torn down, which is what lets the sidebar
picker reveal one mid-playback without a rerun.
"""

from __future__ import annotations

import hashlib
import html as _html
import json
import pathlib
from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from behavior_graphs.envs.trajectory_dir import DEFAULT_CAMERA
from behavior_graphs.viz.media import POSTER_WIDTH, poster_src, video_src
from behavior_graphs.viz.plotly.clusters import CLUSTER_COLORS
from behavior_graphs.viz.streamlit.components.camera_views import (
    channel_name as camera_channel_name,
)


def slice_indicator(
    slice_start: int,
    slice_end: int,
    total_frames: int,
    key: str = "slice_ind",
) -> go.Figure:
    pct = (slice_end - slice_start) / total_frames * 100 if total_frames > 0 else 0.0
    label = f"slice: frames {slice_start}–{slice_end} ({pct:.0f}%)"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[total_frames],
            y=[""],
            orientation="h",
            marker_color="lightgray",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[slice_end - slice_start],
            y=[""],
            orientation="h",
            base=slice_start,
            marker_color="#e87722",
            text=label,
            textposition="inside",
            insidetextanchor="middle",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        barmode="overlay",
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, total_frames], title=None),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    return fig


def cluster_timeline(
    per_frame_labels: np.ndarray,
    cluster_colors: list[str],
    fps: int = 10,
    height: int = 80,
    key: str = "clust_tl",
    cluster_names: dict | None = None,
) -> go.Figure:
    """Multi-colored timeline bar showing cluster assignment across an episode.

    Args:
        per_frame_labels: Integer cluster label per frame, shape (T,). -1 = unlabeled.
        cluster_colors: Color palette indexed by cluster id.
        fps: Frames-per-second, used to format hover labels.
        height: Plotly figure height in pixels.
        key: Streamlit widget key.
        cluster_names: Optional {cluster_id: name} mapping for hover text.

    Returns the Plotly figure (also renders it).
    """
    if len(per_frame_labels) == 0:
        return go.Figure()

    # Run-length encode to get contiguous segments
    labels = np.asarray(per_frame_labels, dtype=np.int64)
    total_frames = len(labels)
    segments: list[tuple[int, int, int]] = []  # (start, end_excl, label)
    i = 0
    while i < total_frames:
        j = i + 1
        while j < total_frames and labels[j] == labels[i]:
            j += 1
        segments.append((i, j, int(labels[i])))
        i = j

    fig = go.Figure()
    n_colors = len(cluster_colors)
    for seg_start, seg_end, cid in segments:
        color = "#555555" if cid < 0 else cluster_colors[cid % n_colors]
        name = (cluster_names or {}).get(cid, f"Cluster {cid}") if cid >= 0 else "Unlabeled"
        width = seg_end - seg_start
        hover = (
            f"<b>{name}</b><br>"
            f"Frames {seg_start}–{seg_end - 1}<br>"
            f"Time {seg_start / fps:.1f}s–{(seg_end - 1) / fps:.1f}s"
        )
        fig.add_trace(
            go.Bar(
                x=[width],
                y=[""],
                base=seg_start,
                orientation="h",
                marker_color=color,
                marker_line_width=0,
                showlegend=False,
                hovertemplate=hover + "<extra></extra>",
                name=name,
            )
        )

    fig.update_layout(
        barmode="overlay",
        height=height,
        margin=dict(l=0, r=0, t=0, b=20),
        xaxis=dict(range=[0, total_frames], title="frame", tickfont=dict(size=10)),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.9)", font_color="white"),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    return fig


def start_frame(
    slice_start: int | None, segments: list | None, seek_frame: int | None = None
) -> int | None:
    """Which frame the player should open on, or None to start at the top.

    A player that highlights part of an episode should open there — otherwise
    you click a behavior and watch the arm do something else until the segment
    finally arrives. ``slice_start`` names it directly; in multi-segment mode
    (a path, or a src→tgt transition) the earliest segment is the one the
    selection starts at.

    ``seek_frame`` overrides both, for a selection whose highlight and whose
    interesting moment are not the same frame: a terminal leaf highlights the
    whole path that reached the outcome, but what you clicked it to watch is the
    last behavior before it, so playback opens there and the earlier segments
    stay on the timeline as context.
    """
    if seek_frame is not None:
        return int(seek_frame)
    if slice_start is not None:
        return int(slice_start)
    starts = [int(seg[0]) for seg in (segments or [])]
    return min(starts) if starts else None


def _seg_overlay(uid: str, i: int, *, badge: str) -> str:
    """One view's behavior-segment border + badge, both driven from JS.

    Per view rather than once per player: with several cameras on screen the
    "this behavior is happening now" cue has to be on each of them, otherwise it
    reads as a property of whichever panel happens to carry it.

    An empty *badge* starts out hidden — the multi-segment mode names the
    behavior from JS once one is under the playhead, and a styled empty pill
    before then is a floating red dot.
    """
    badge_hidden = not badge
    return (
        f'<div id="segb_{uid}_{i}" style="position:absolute;top:0;left:0;width:100%;height:100%;'
        "pointer-events:none;border-radius:4px;border:3px solid transparent;"
        'box-sizing:border-box;transition:border-color 0.1s,box-shadow 0.1s;"></div>'
        f'<div id="segbadge_{uid}_{i}" style="position:absolute;top:8px;right:8px;'
        "font-size:10px;font-weight:700;color:#fff;"
        f"{'background:#d62728;' if badge else ''}"
        "padding:2px 8px;border-radius:10px;pointer-events:none;"
        f"{'display:none;' if badge_hidden else ''}"
        f'transition:background 0.1s;">{badge}</div>'
    )


def mp4_player(
    video_path: str | pathlib.Path | None = None,
    label: str = "",
    slice_start: int | None = None,
    slice_end: int | None = None,
    total_frames: int | None = None,
    key: str = "mp4p",
    max_height_px: int | None = None,
    fps: int = 10,
    per_frame_labels: np.ndarray | None = None,
    cluster_colors: list[str] | None = None,
    cluster_names: dict | None = None,
    slice2_start: int | None = None,
    slice2_end: int | None = None,
    bar1_label: str = "",
    bar2_label: str = "",
    segments: list | None = None,  # [(start_frame, end_frame, label, color), ...]
    seek_frame: int | None = None,
    videos: Sequence[tuple[str, str | pathlib.Path]] | None = None,
    visible_cameras: Sequence[str] | None = None,
    view_transforms: Mapping[str, Mapping[str, object]] | None = None,
    show_segments: bool = True,
    show_camera_labels: bool = True,
    seek_from_end_s: float | None = None,
    gt_segments: Sequence[tuple[float, float, str, bool]] | None = None,
    gt_duration_sec: float | None = None,
) -> None:
    """One episode's video, with the panel's timelines and overlays around it.

    Pass ``video_path`` for a single view, or ``videos`` as ``[(camera, path)]``
    for a multi-camera episode. Every entry in *videos* is rendered side by side
    and kept in sync; *visible_cameras* (default: all of them) picks which are on
    screen, and a message on the sidebar picker's channel
    (:data:`~behavior_graphs.viz.streamlit.components.camera_views.CHANNEL`)
    re-picks them later without a
    rerun. All the frame-indexed arguments — slices, ``per_frame_labels``,
    ``segments`` — describe the episode, so they apply to every view.

    ``seek_frame`` says which frame to open on when that is not where the
    highlight starts (see :func:`start_frame`); by default the player opens on
    the slice, or on the earliest segment.

    ``show_segments`` off drops every behavior annotation — the timeline bars,
    the badge and border on each view, the loop-segment button — while leaving
    the opening frame where the annotation would have put it. It is for a
    caller that wants the rollout watched, not the labelling read.
    ``show_camera_labels`` off drops the per-view camera name badge.

    ``seek_from_end_s`` opens the clip that many seconds before its end. In
    seconds and applied in the browser, not converted to a frame here: these
    clips are recorded at wall-clock speed but stepped at a few Hz, so
    ``total_frames``/``fps`` disagrees with the real duration and "six seconds
    from the end" is only exact against ``v.duration``.

    *view_transforms* is ``{camera: {"rotate": deg, "mirror": bool}}`` from
    ``configs/data/`` (see
    :func:`behavior_graphs.viz.streamlit.data.camera_transforms`) for cameras whose
    mounting does not match how their frames are stored. It is baked into the web
    proxy and the poster, not applied to the ``<video>`` — the source clips are
    still never rewritten, but the element (and so its control bar) stays
    upright.
    """
    if label:
        st.caption(label)

    if videos is not None:
        # An empty list is a legitimate answer from episode_camera_clips (no
        # media dir at all), so it renders the same "no video" state as a clip
        # that turns out to be unservable — not an exception that would take the
        # whole page down around a panel that used to degrade on its own.
        sources = [(camera, pathlib.Path(path)) for camera, path in videos]
    elif video_path is not None:
        sources = [(DEFAULT_CAMERA, pathlib.Path(video_path))]
    else:
        raise ValueError("mp4_player needs either video_path or videos")

    vid_h = max_height_px if max_height_px else 400
    seek_frame = start_frame(slice_start, segments, seek_frame)
    # After start_frame, so hiding the annotations does not also move the
    # playhead back to frame 0 -- where it opens is not part of the labelling.
    if not show_segments:
        slice_start = slice_end = slice2_start = slice2_end = None
        segments = None
        per_frame_labels = None
    uid = "v" + hashlib.md5(("|".join(str(p) for _, p in sources) + key).encode()).hexdigest()[:8]
    # Serve a web-sized proxy over Streamlit's /media endpoint instead of
    # base64-inlining the raw file: the browser caches the body and every
    # rerun re-sends a URL rather than tens of megabytes of websocket frame.
    # One coordinate per view so the media manager releases the right file when
    # this player switches episodes.
    views: list[dict[str, str]] = []
    for camera, path in sources:
        # A configured rotation is baked into the proxy and poster rather than
        # applied as a CSS transform here: transforming the <video> turns its
        # native control bar upside down with the picture, and leaves every
        # frame decoded from the clip (poster, strip tiles) the wrong way up.
        xform = (view_transforms or {}).get(camera)
        src_url = video_src(path, coordinates=f"mp4p:{key}:{camera}", transform=xform)
        if src_url is None:
            continue
        poster_url = poster_src(
            path,
            coordinates=f"mp4p-poster:{key}:{camera}",
            width=POSTER_WIDTH,
            transform=xform,
        )
        views.append(
            {
                "camera": camera,
                "src": src_url,
                "poster": f' poster="{poster_url}"' if poster_url else "",
            }
        )
    if not views:
        # The clips are gone (a cached episode index can outlive a deleted
        # rollout) or too big to inline without a runtime — say so rather than
        # emitting a <video> with an empty src.
        named = f": `{sources[0][1].name}`" if sources else ""
        st.info(f"Video unavailable{named}")
        return
    # Fall back to the first view rather than an empty grid: a selection that
    # names only views this episode is missing would otherwise render nothing.
    wanted = {v["camera"] for v in views} if visible_cameras is None else set(visible_cameras)
    if not wanted.intersection(v["camera"] for v in views):
        wanted = {views[0]["camera"]}
    total_s = (total_frames / fps) if (total_frames and fps > 0) else None
    # Normalisation factor for bar positions: use total_frames (policy steps)
    # rather than fps*video_duration — the two differ for VLA/low-Hz robots
    # whose videos are recorded at wall-clock speed but stepped at ~1–5 Hz.
    _nf = total_frames if (total_frames and total_frames > 0) else 0
    extra_h = 4

    # ── Cluster timeline with live cursor ────────────────────────────────────
    cluster_tl_html = ""
    cluster_tl_js = ""
    if (
        per_frame_labels is not None
        and cluster_colors is not None
        and total_s is not None
        and len(per_frame_labels) > 0
    ):
        labels_arr = np.asarray(per_frame_labels, dtype=np.int64)
        n_total = len(labels_arr)
        segs: list[tuple[int, int, int]] = []
        i = 0
        while i < n_total:
            j = i + 1
            while j < n_total and labels_arr[j] == labels_arr[i]:
                j += 1
            segs.append((i, j, int(labels_arr[i])))
            i = j

        n_c = len(cluster_colors)
        segs_html = ""
        segs_data = []
        for s0, s1, cid in segs:
            color = "#444" if cid < 0 else cluster_colors[cid % n_c]
            name = (cluster_names or {}).get(cid, f"Cluster {cid}") if cid >= 0 else "Unlabeled"
            left_pct = s0 / n_total * 100
            w_pct = (s1 - s0) / n_total * 100
            segs_html += (
                f'<div style="position:absolute;left:{left_pct:.3f}%;width:{w_pct:.3f}%;'
                f'height:100%;background:{color};"></div>'
            )
            segs_data.append({"l": s0 / n_total, "r": s1 / n_total, "n": name})
        segs_json = json.dumps(segs_data)

        cluster_tl_html = f"""
<div style="margin-top:6px;position:relative;">
  <div id="ctlt_{uid}" style="position:absolute;bottom:22px;
       background:rgba(20,20,20,0.92);color:#fff;font-size:11px;padding:3px 8px;
       border-radius:4px;pointer-events:none;white-space:nowrap;display:none;
       transform:translateX(-50%);z-index:999;"></div>
  <div id="ctl_{uid}" style="position:relative;height:14px;border-radius:3px;
       overflow:hidden;cursor:pointer;background:#222;">
    {segs_html}
    <div id="ctlc_{uid}" style="position:absolute;top:0;left:0%;width:2px;height:100%;
         background:rgba(255,255,255,0.85);pointer-events:none;"></div>
  </div>
  <div style="font-size:10px;color:#666;margin-top:4px;">cluster timeline</div>
</div>"""

        cluster_tl_js = f"""
  var ctl=document.getElementById('ctl_{uid}');
  var ctlc=document.getElementById('ctlc_{uid}');
  var ctlt=document.getElementById('ctlt_{uid}');
  var segs={segs_json};
  if(ctl&&ctlc){{
    v.addEventListener('timeupdate',function(){{
      var dur=v.duration||{total_s:.3f};
      ctlc.style.left=(Math.min(1,v.currentTime/dur)*100).toFixed(2)+'%';
    }});
    ctl.addEventListener('click',function(e){{
      var dur=v.duration||{total_s:.3f};
      var r=ctl.getBoundingClientRect();
      v.currentTime=(e.clientX-r.left)/r.width*dur;
    }});
    ctl.addEventListener('mousemove',function(e){{
      if(!ctlt)return;
      var r=ctl.getBoundingClientRect();
      var frac=(e.clientX-r.left)/r.width;
      for(var i=0;i<segs.length;i++){{
        if(frac>=segs[i].l&&frac<segs[i].r){{
          ctlt.textContent=segs[i].n;
          ctlt.style.left=((e.clientX-r.left)/r.width*100).toFixed(1)+'%';
          ctlt.style.display='block';
          return;
        }}
      }}
      ctlt.style.display='none';
    }});
    ctl.addEventListener('mouseleave',function(){{if(ctlt)ctlt.style.display='none';}});
  }}"""
        extra_h += 40

    # -- Ground-truth subtask strip ------------------------------------------
    # Below the cluster timeline and laid out the same way, so the two read as
    # what they are: one episode cut by a human and by the clustering.
    #
    # Seconds, not frames, unlike the cluster strip above. The annotation is
    # authored in seconds against the clip, so scaling by the clip's own
    # duration is exact -- converting to policy steps would inherit the
    # frames-vs-wall-clock mismatch the rest of this file works around.
    gt_tl_html = ""
    gt_tl_js = ""
    if gt_segments:
        gt_dur = float(gt_duration_sec or 0.0) or max(float(e) for _, e, _, _ in gt_segments)
        if gt_dur > 0:
            # One colour per distinct subtask, assigned in order of first
            # appearance, so a subtask that recurs is the same colour every
            # time -- which is the point of showing a cyclic task.
            colour_of: dict[str, str] = {}
            for _, _, _name, _is_task in gt_segments:
                if _is_task and _name not in colour_of:
                    colour_of[_name] = CLUSTER_COLORS[len(colour_of) % len(CLUSTER_COLORS)]

            gt_html = ""
            gt_data = []
            for _a, _b, _name, _is_task in gt_segments:
                left = max(0.0, min(1.0, float(_a) / gt_dur)) * 100
                width = max(0.0, min(1.0, (float(_b) - float(_a)) / gt_dur)) * 100
                if width <= 0:
                    continue
                # Non-task spans are what scoring folds into ignore regions.
                # Hatched grey rather than omitted: a gap would read as "not
                # annotated" when the truth is "annotated and excluded".
                if _is_task:
                    style = f"background:{colour_of.get(_name, '#888')};"
                else:
                    style = (
                        "background:repeating-linear-gradient(45deg,#555,#555 3px,"
                        "#3a3a3a 3px,#3a3a3a 6px);"
                    )
                gt_html += (
                    f'<div style="position:absolute;left:{left:.3f}%;width:{width:.3f}%;'
                    f'height:100%;{style}"></div>'
                )
                gt_data.append(
                    {
                        "l": float(_a) / gt_dur,
                        "r": float(_b) / gt_dur,
                        "n": _name if _is_task else f"{_name} (excluded from scoring)",
                    }
                )
            n_task = sum(1 for _, _, _, t in gt_segments if t)
            n_skip = len(gt_segments) - n_task
            caption = f"ground truth: {n_task} subtask" + ("s" if n_task != 1 else "")
            if n_skip:
                caption += f", {n_skip} excluded"
            gt_tl_html = f"""
<div style="margin-top:6px;position:relative;">
  <div id="gtt_{uid}" style="position:absolute;bottom:22px;
       background:rgba(20,20,20,0.92);color:#fff;font-size:11px;padding:3px 8px;
       border-radius:4px;pointer-events:none;white-space:nowrap;display:none;
       transform:translateX(-50%);z-index:999;max-width:90%;overflow:hidden;
       text-overflow:ellipsis;"></div>
  <div id="gt_{uid}" style="position:relative;height:14px;border-radius:3px;
       overflow:hidden;cursor:pointer;background:#222;">
    {gt_html}
    <div id="gtc_{uid}" style="position:absolute;top:0;left:0%;width:2px;height:100%;
         background:rgba(255,255,255,0.85);pointer-events:none;"></div>
  </div>
  <div style="font-size:10px;color:#666;margin-top:4px;">{caption}</div>
</div>"""
            gt_tl_js = f"""
  var gtl=document.getElementById('gt_{uid}');
  var gtlc=document.getElementById('gtc_{uid}');
  var gtlt=document.getElementById('gtt_{uid}');
  var gsegs={json.dumps(gt_data)};
  if(gtl&&gtlc){{
    v.addEventListener('timeupdate',function(){{
      var dur=v.duration||{gt_dur:.3f};
      gtlc.style.left=(Math.min(1,v.currentTime/dur)*100).toFixed(2)+'%';
    }});
    gtl.addEventListener('click',function(e){{
      var dur=v.duration||{gt_dur:.3f};
      var r=gtl.getBoundingClientRect();
      v.currentTime=(e.clientX-r.left)/r.width*dur;
    }});
    gtl.addEventListener('mousemove',function(e){{
      if(!gtlt)return;
      var r=gtl.getBoundingClientRect();
      var frac=(e.clientX-r.left)/r.width;
      for(var i=0;i<gsegs.length;i++){{
        if(frac>=gsegs[i].l&&frac<gsegs[i].r){{
          gtlt.textContent=gsegs[i].n;
          gtlt.style.left=(frac*100).toFixed(1)+'%';
          gtlt.style.display='block';
          return;
        }}
      }}
      gtlt.style.display='none';
    }});
    gtl.addEventListener('mouseleave',function(){{if(gtlt)gtlt.style.display='none';}});
  }}"""
            extra_h += 40

    # ── Behavior slice bar with playhead ─────────────────────────────────────
    # Bar positions and playhead are computed dynamically in JS from v.duration so they always
    # match the actual video length regardless of any mismatch between total_frames/fps and the
    # codec-reported duration.
    if slice_start is not None and slice_end is not None:
        start_s = f"{slice_start / fps:.1f}s"
        end_s = f"{slice_end / fps:.1f}s"
        total_s_fb = f"{total_s:.3f}" if total_s else "0"

        has_bar2 = slice2_start is not None and slice2_end is not None and slice2_end > slice2_start
        if has_bar2:
            start_s2 = f"{slice2_start / fps:.1f}s"
            end_s2 = f"{slice2_end / fps:.1f}s"
            bar2_div = (
                f'<div id="bar2_{uid}" style="position:absolute;left:0%;width:0%;height:100%;'
                f'background:#38bdf8;border-radius:5px;opacity:0.85;"></div>'
            )
            lbl2 = f" {bar2_label}" if bar2_label else ""
            legend2_html = (
                f'&nbsp;|&nbsp;<span style="color:#38bdf8;">■</span>'
                f'<span style="color:#aaa;">{lbl2} {start_s2}–{end_s2}</span>'
            )
            bar2_js = f"""
  if(bar2){{
    bar2.style.left=Math.max(0,_pct({slice2_start},dur)).toFixed(2)+'%';
    bar2.style.width=Math.max(0.5,_pct({slice2_end},dur)-_pct({slice2_start},dur)).toFixed(2)+'%';
  }}"""
        else:
            bar2_div = ""
            legend2_html = ""
            bar2_js = ""

        lbl1 = f" {bar1_label}" if bar1_label else ""
        # "plays from start" was left behind when the player learned to open on the
        # highlighted span: it is the one line on screen claiming the playhead is at
        # frame 0, next to a playhead that visibly is not. Say where it does open.
        opens_at = (
            "▶ plays from the start" if not seek_frame else f"▶ opens at {seek_frame / fps:.1f}s"
        )
        timeline = f"""
        <div id="tl_{uid}" style="margin-top:6px;position:relative;height:10px;
             background:#333;border-radius:5px;overflow:hidden;cursor:pointer;">
          <div id="bar1_{uid}" style="position:absolute;left:0%;width:0%;height:100%;
                      background:#f5a623;opacity:0.85;"></div>
          {bar2_div}
          <div id="ph_{uid}" style="position:absolute;top:-3px;left:0%;width:3px;height:16px;
               background:#fff;border-radius:2px;transition:left 0.1s linear;"></div>
        </div>
        <div style="font-size:10px;color:#888;margin-top:3px;">
          {opens_at} &nbsp;|&nbsp;<span style="color:#f5a623;">■</span><span style="color:#aaa;">{lbl1} {start_s}–{end_s}</span>{legend2_html}
        </div>"""
        playhead_js = f"""
  var ph=document.getElementById('ph_{uid}');
  var tl=document.getElementById('tl_{uid}');
  var bar1=document.getElementById('bar1_{uid}');
  var bar2=document.getElementById('bar2_{uid}');
  function updateBars(){{
    var dur=v.duration||{total_s_fb};
    if(!dur||!isFinite(dur)) return;
    if(bar1){{
      bar1.style.left=Math.max(0,_pct({slice_start},dur)).toFixed(2)+'%';
      bar1.style.width=Math.max(0.5,_pct({slice_end},dur)-_pct({slice_start},dur)).toFixed(2)+'%';
    }}{bar2_js}
  }}
  v.addEventListener('loadedmetadata',updateBars);
  if(v.readyState>=1) updateBars();
  v.addEventListener('timeupdate',function(){{
    var dur=v.duration||{total_s_fb};
    var pct=v.currentTime/dur*100;
    if(ph) ph.style.left=Math.min(100,pct).toFixed(1)+'%';
  }});
  tl.addEventListener('click',function(e){{
    var dur=v.duration||{total_s_fb};
    var rect=tl.getBoundingClientRect();
    var pct=(e.clientX-rect.left)/rect.width;
    v.currentTime=pct*dur;
  }});"""
        extra_h += 38

        # ── Segment glow overlay (border + badge on each view) ─────────────────
        seg_badge = "behavior inactive"
        highlight_js = f"""
  function _hl(){{
    var dur=v.duration||{total_s_fb};
    var ins=v.currentTime+_EPS>=_sec({slice_start},dur)&&v.currentTime<=_sec({slice_end},dur);
    for(var i=0;i<segbs.length;i++){{
      if(segbs[i]){{segbs[i].style.borderColor=ins?'#2ca02c':'#d62728';
                    segbs[i].style.boxShadow=ins?'0 0 0 3px rgba(44,160,44,0.35)'
                                                :'0 0 0 3px rgba(214,39,40,0.35)';}}
      if(segbadges[i]){{segbadges[i].style.background=ins?'#2ca02c':'#d62728';
                        segbadges[i].textContent=ins?'behavior active':'behavior inactive';}}
    }}
  }}
  _hl();
  v.addEventListener('loadedmetadata',_hl);
  v.addEventListener('timeupdate',_hl);"""

        # ── Loop-segment button ───────────────────────────────────────────────
        loop_btn_html = (
            f'<div style="margin-top:4px;">'
            f'<button id="lbtn_{uid}" title="Loop over the annotated segment only" '
            f'style="font-size:10px;padding:2px 8px;border-radius:4px;'
            f'border:1px solid #555;background:#222;color:#999;cursor:pointer;">'
            f"⟳ Loop segment</button></div>"
        )
        loop_js = f"""
  var _lp_{uid}=true;
  var lbtn=document.getElementById('lbtn_{uid}');
  if(lbtn){{
    lbtn.style.color='#2ca02c';lbtn.style.borderColor='#2ca02c';
    lbtn.addEventListener('click',function(){{
      _lp_{uid}=!_lp_{uid};
      lbtn.style.color=_lp_{uid}?'#2ca02c':'#999';
      lbtn.style.borderColor=_lp_{uid}?'#2ca02c':'#555';
      if(_lp_{uid}){{var _d=v.duration||{total_s_fb};v.currentTime=_sec({slice_start},_d);v.play();}}
    }});
  }}
  v.addEventListener('timeupdate',function(){{
    var _d2=v.duration||{total_s_fb};
    if(_lp_{uid}&&v.currentTime>=_sec({slice_end},_d2)){{v.currentTime=_sec({slice_start},_d2);}}
  }});"""
        extra_h += 28
    elif segments:
        # ── Multi-segment mode: colored bars + dynamic behavior-name badge ─────
        total_s_fb = f"{total_s:.3f}" if total_s else "0"
        bars_html = ""
        segs_json_data = []
        for seg_s, seg_e, seg_lbl, seg_col in segments:
            bars_html += (
                f'<div id="msbar_{uid}_{seg_s}" '
                f'style="position:absolute;left:0%;width:0%;height:100%;'
                f'background:{seg_col};border-radius:5px;opacity:0.85;"></div>'
            )
            segs_json_data.append({"s": seg_s, "e": seg_e, "l": seg_lbl, "c": seg_col})
        segs_json = json.dumps(segs_json_data)

        timeline = f"""
        <div id="tl_{uid}" style="margin-top:6px;position:relative;height:10px;
             background:#333;border-radius:5px;cursor:pointer;">
          {bars_html}
          <div id="ph_{uid}" style="position:absolute;top:-3px;left:0%;width:3px;height:16px;
               background:#fff;border-radius:2px;transition:left 0.1s linear;"></div>
        </div>
        <div style="font-size:10px;color:#888;margin-top:3px;">{
            "  |  ".join(f'<span style="color:{c};">■</span> {lab}' for _, _, lab, c in segments)
        }</div>"""

        seg_badge = ""  # named from JS once a segment is under the playhead

        playhead_js = f"""
  var ph=document.getElementById('ph_{uid}');
  var tl=document.getElementById('tl_{uid}');
  var _msegs={segs_json};
  function updateMsBars(){{
    var dur=v.duration||{total_s_fb};
    if(!dur||!isFinite(dur)) return;
    for(var i=0;i<_msegs.length;i++){{
      var b=document.getElementById('msbar_{uid}_'+_msegs[i].s);
      if(b){{
        b.style.left=Math.max(0,_pct(_msegs[i].s,dur)).toFixed(2)+'%';
        b.style.width=Math.max(0.5,_pct(_msegs[i].e,dur)-_pct(_msegs[i].s,dur)).toFixed(2)+'%';
      }}
    }}
  }}
  v.addEventListener('loadedmetadata',updateMsBars);
  if(v.readyState>=1) updateMsBars();
  v.addEventListener('timeupdate',function(){{
    var dur=v.duration||{total_s_fb};
    var pct=v.currentTime/dur*100;
    if(ph) ph.style.left=Math.min(100,pct).toFixed(1)+'%';
  }});
  tl.addEventListener('click',function(e){{
    var dur=v.duration||{total_s_fb};
    var rect=tl.getBoundingClientRect();
    v.currentTime=(e.clientX-rect.left)/rect.width*dur;
  }});"""

        highlight_js = f"""
  var _msegs2={segs_json};
  function _mhl(){{
    var dur=v.duration||{total_s_fb};
    var ct=v.currentTime;
    var found=null;
    // Half-open, except at the very end: the panels close the gaps between
    // segments, so each one's end IS the next one's start, and testing both ends
    // inclusively made a boundary frame match the segment that just finished.
    // That is exactly the frame a terminal selection opens on (see start_frame),
    // so the badge named the previous behavior for as long as the video sat
    // paused. The last segment keeps its end, or the final frame matches nothing.
    var cte=ct+_EPS;
    for(var i=0;i<_msegs2.length;i++){{
      var _s=_sec(_msegs2[i].s,dur), _e=_sec(_msegs2[i].e,dur);
      if(cte>=_s&&(cte<_e||(i===_msegs2.length-1&&ct<=_e))){{found=_msegs2[i];break;}}
    }}
    for(var j=0;j<segbs.length;j++){{
      if(found){{
        if(segbs[j]){{segbs[j].style.borderColor=found.c;
                      segbs[j].style.boxShadow='0 0 0 3px '+found.c+'55';}}
        if(segbadges[j]){{segbadges[j].style.background=found.c;
                          segbadges[j].textContent=found.l;
                          segbadges[j].style.display='';}}
      }}else{{
        if(segbs[j]){{segbs[j].style.borderColor='transparent';segbs[j].style.boxShadow='none';}}
        if(segbadges[j]) segbadges[j].style.display='none';
      }}
    }}
  }}
  _mhl();
  v.addEventListener('loadedmetadata',_mhl);
  v.addEventListener('timeupdate',_mhl);"""

        loop_btn_html = ""
        loop_js = ""
        extra_h += 38
    else:
        timeline = ""
        playhead_js = ""
        seg_badge = None  # no timeline, so no segment overlay at all
        highlight_js = ""
        loop_btn_html = ""
        loop_js = ""

    # ── The view grid ────────────────────────────────────────────────────────
    # One flex row, so the panel's height does not depend on how many views are
    # on screen: hiding one widens the rest instead of reflowing the iframe,
    # which Streamlit sizes from Python and cannot resize from in here.
    show_names = len(views) > 1 and show_camera_labels
    grid_html = ""
    for i, view in enumerate(views):
        camera = _html.escape(view["camera"], quote=True)
        poster_attr = view["poster"]
        src_url = view["src"]
        hidden = "" if view["camera"] in wanted else "display:none;"
        overlay_html = _seg_overlay(uid, i, badge=seg_badge) if seg_badge is not None else ""
        name_badge = (
            f'<div style="position:absolute;top:8px;left:8px;font-size:10px;font-weight:600;'
            f"color:#fff;background:rgba(0,0,0,0.55);padding:2px 6px;border-radius:8px;"
            f"pointer-events:none;max-width:calc(100% - 100px);overflow:hidden;"
            f'text-overflow:ellipsis;white-space:nowrap;">{camera}</div>'
            if show_names
            else ""
        )
        grid_html += f"""
<div id="cam_{uid}_{i}" data-cam="{camera}"
     style="flex:1 1 0;min-width:0;position:relative;{hidden}">
<!-- Fixed height, not max-height: a <video> has no intrinsic size until its
     metadata arrives, so it laid out at zero and then jumped. -->
<video id="{uid}_{i}" controls preload="metadata" playsinline{poster_attr}
  style="width:100%;height:{vid_h}px;object-fit:contain;border-radius:4px;
         background:#000;display:block;">
  <source src="{src_url}" type="video/mp4">
</video>
{overlay_html}
{name_badge}
</div>"""

    html = f"""
<div id="grid_{uid}" style="display:flex;gap:6px;width:100%;">{grid_html}
</div>
{cluster_tl_html}
{gt_tl_html}
{timeline}
{loop_btn_html}
<script>
(function(){{
  var vids=[];
  for(var i=0;i<{len(views)};i++){{
    var el=document.getElementById('{uid}_'+i);
    if(el) vids.push(el);
  }}
  if(!vids.length) return;
  // Everything below the grid — playhead, cluster strip, loop — reads one
  // clock. vids[0] is always a valid one: views are kept in sync whether or not
  // they are on screen, precisely so that hiding one changes nothing but CSS.
  var v=vids[0];
  var segbs=[],segbadges=[];
  for(var i=0;i<{len(views)};i++){{
    segbs.push(document.getElementById('segb_{uid}_'+i));
    segbadges.push(document.getElementById('segbadge_{uid}_'+i));
  }}
  var _nf={_nf};
  function _pct(t,dur){{return _nf>0?t/_nf*100:t/{fps}/dur*100;}}
  // A clip's real duration and total_frames/fps disagree for VLA/low-Hz robots
  // (recorded at wall-clock speed, stepped at ~1-5 Hz), so frame -> seconds
  // goes through the duration. Falls back to fps only when there is no usable
  // duration to scale against, which is also what _pct does.
  function _sec(t,dur){{
    return (_nf>0&&dur>0&&isFinite(dur))?t/_nf*dur:t/{fps};
  }}
  // Assigning currentTime snaps to the clip's own frame grid, which can land a
  // frame *before* the one asked for — so a player that opens exactly on a
  // behavior's first frame reports a time just short of it, and every "is a
  // behavior under the playhead" test below decided the answer was the behavior
  // that had already finished. Compared against a time nudged forward by this
  // much instead: far under the shortest behavior anyone can see, far over the
  // frame interval of any clip.
  var _EPS=0.15;
  var _seekF={seek_frame if seek_frame is not None else -1};
{_sync_js(uid, len(views))}
  // Seek through _sec, not slice_start/fps: the bars below are positioned that
  // way, and a playhead on a different scale lands outside its own bar. Defined
  // after the sync handlers so the opening seek carries the other views with it
  // instead of leaving them parked on frame 0 until something plays.
  var _fromEnd={seek_from_end_s if seek_from_end_s is not None else -1};
  function seek(){{
    if(_fromEnd>=0){{
      var d=v.duration;
      if(isFinite(d)&&d>0){{v.currentTime=Math.max(0,d-_fromEnd);return;}}
    }}
    if(_seekF<0) return;
    var t=_sec(_seekF, v.duration||{total_s if total_s else 0});
    if(isFinite(t)&&t>0) v.currentTime=t;
  }}
  v.readyState>=1 ? seek() : v.addEventListener('loadedmetadata',seek,{{once:true}});
  {cluster_tl_js}
  {gt_tl_js}
  {playhead_js}
  {highlight_js}
  {loop_js}
}})();
</script>"""

    components.html(html, height=vid_h + extra_h, scrolling=False)


def _sync_js(uid: str, n_views: int) -> str:
    """Keep the views frame-aligned, and let the sidebar re-pick which are shown.

    Peer-to-peer rather than one master driving followers: the user can hit play
    or drag the scrubber on *any* of the views, and whichever they touched is
    the one the others follow.

    Two things stop an event round from echoing forever. Play, pause and rate
    are conditional (only play if paused, only assign a rate that differs), so a
    propagated event finds nothing left to do. Seeks can't rely on that, and a
    reentrancy flag doesn't help because media events are queued rather than
    dispatched synchronously: a seek takes long enough to complete that the
    reference view has moved on by the time ``seeked`` fires on the target, so
    the echo would drag the reference *backwards* by that much, and the two would
    trade seeks for as long as playback lasted. Propagated seeks are therefore
    marked and their ``seeked`` swallowed.
    """
    if n_views < 2:
        # Single view: no peers to sync, and no panel the picker could reveal.
        return ""
    return f"""
  // ── Keep the views together ─────────────────────────────────────────────
  function _at(o,src){{
    // Views of one episode are the same length, so absolute time is the
    // honest mapping; fall back to the same *fraction* when a camera was
    // recorded over a different span, which beats drifting off the end.
    var d=o.duration,s=src.duration;
    if(d>0&&s>0&&isFinite(d)&&isFinite(s)&&Math.abs(d-s)>0.5) return d*(src.currentTime/s);
    return src.currentTime;
  }}
  var _busy=false;
  function _peers(src,fn){{
    if(_busy) return;
    _busy=true;
    try{{ for(var i=0;i<vids.length;i++){{ if(vids[i]!==src) fn(vids[i]); }} }}
    finally{{ _busy=false; }}
  }}
  function _align(o,src,tol){{
    var t=_at(o,src);
    // Clamp to what this view can actually reach. Views within 0.5s of each
    // other are aligned on absolute time, so the longer one runs past the end of
    // the shorter: assigning 10.3 to a 10.0s clip lands on 10.0, the 'seeked'
    // that raises does not match _pgSeek, so it reads as the user scrubbing and
    // drags the reference *back* to 10.0 — the pair then trade seeks over the
    // last fraction of a second and the longer view never reaches 'ended'.
    if(o.duration>0&&isFinite(o.duration)&&t>o.duration) t=o.duration;
    if(Math.abs(o.currentTime-t)>tol){{
      // Remember where we sent it, so the 'seeked' this raises is not mistaken
      // for the user scrubbing this view — which would seek everyone else back
      // to where the seek *started* rather than where the source is now. The
      // position, not a bare flag: an assignment before metadata arrives raises
      // no event at all, and a flag left set would swallow the user's next seek.
      o._pgSeek=t;
      o.currentTime=t;
    }}
  }}
  vids.forEach(function(s){{
    s.addEventListener('play',function(){{_peers(s,function(o){{
      _align(o,s,0.08); if(o.paused) o.play().catch(function(){{}});
    }});}});
    s.addEventListener('pause',function(){{_peers(s,function(o){{
      if(!o.paused) o.pause(); _align(o,s,0.08);
    }});}});
    s.addEventListener('seeked',function(){{
      var sent=s._pgSeek;
      s._pgSeek=null;
      if(sent!=null&&Math.abs(s.currentTime-sent)<0.05) return;  // our own seek
      _peers(s,function(o){{_align(o,s,0.08);}});
    }});
    s.addEventListener('ratechange',function(){{_peers(s,function(o){{
      if(o.playbackRate!==s.playbackRate) o.playbackRate=s.playbackRate;
    }});}});
    // A view whose metadata lands late cannot be seeked before it does, so it
    // catches up here rather than sitting on frame 0 until playback starts.
    s.addEventListener('loadedmetadata',function(){{ if(s!==v) _align(s,v,0.08); }});
  }});
  // Decoders run on their own clocks, so correct accumulated drift from the
  // reference view. Looser tolerance than a seek: correcting every frame would
  // stutter playback for no visible gain.
  v.addEventListener('timeupdate',function(){{_peers(v,function(o){{_align(o,v,0.25);}});}});

  // ── Which views are on screen ───────────────────────────────────────────
  // The sidebar picker reruns its own fragment only, so the players are never
  // re-rendered; it reaches them here instead. Component iframes are srcdoc
  // with allow-same-origin, so they share the app's origin and this channel.
  function _show(names){{
    var wanted={{}};
    for(var i=0;i<names.length;i++) wanted[names[i]]=1;
    var boxes=[],on=[];
    for(var i=0;i<{n_views};i++){{
      var box=document.getElementById('cam_{uid}_'+i);
      if(!box) continue;
      boxes.push(box); on.push(!!wanted[box.getAttribute('data-cam')]);
    }}
    if(on.indexOf(true)<0) on[0]=true;  // never leave an empty grid
    for(var i=0;i<boxes.length;i++){{
      var revealed=on[i]&&boxes[i].style.display==='none';
      boxes[i].style.display=on[i]?'':'none';
      if(revealed&&vids[i]&&vids[i]!==v) _align(vids[i],v,0.08);
    }}
  }}
  try{{
    var _bus=new BroadcastChannel({json.dumps(camera_channel_name())});
    _bus.onmessage=function(e){{
      if(e&&e.data&&Array.isArray(e.data.cameras)) _show(e.data.cameras);
    }};
  }}catch(err){{}}"""
