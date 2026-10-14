"""A rollout video and its progress curves in ONE iframe, sharing a playhead.

``st.plotly_chart`` and :func:`~...mp4_player.mp4_player` each render into their
own component iframe, and script in one cannot reach the other -- which is why
the moving cursor used to live on a segment bar under the video while the curves
below sat still. Putting the ``<video>`` and the figure in the same document
removes the wall: a ``timeupdate`` listener can call ``Plotly.relayout`` on a
vertical line shape, and a click on the plot can seek the video.

plotly.js is loaded from the CDN plotly.py itself versions against, not inlined.
The repo's other embed (pyvis, ``cdn_resources="in_line"``) inlines its library,
but that one is small and this one is 4.85 MB -- per rerun, over the websocket,
against exactly the payload the media-proxy work went to lengths to cut. The
tradeoff is a hard dependency on the browser reaching the CDN, so the panel says
so plainly rather than rendering an empty box if it cannot.
"""

from __future__ import annotations

import pathlib
import uuid

import plotly.graph_objects as go
import plotly.io as pio
import streamlit.components.v1 as components

from behavior_graphs.viz.media import video_src

#: The playhead drawn over the curves. Streamlit's primary red, so it reads as
#: the one live element rather than as another series.
CURSOR_COLOR = "#ff4b4b"


def _cursor_shape() -> dict:
    """A full-height vertical line at x=0, moved from JS by index into shapes."""
    return {
        "type": "line",
        "x0": 0.0,
        "x1": 0.0,
        "y0": 0,
        "y1": 1,
        "yref": "paper",
        "line": {"color": CURSOR_COLOR, "width": 2},
        "layer": "above",
    }


def video_curve_player(
    video_path: str | pathlib.Path,
    fig: go.Figure,
    *,
    key: str,
    video_height_px: int = 300,
    chart_height_px: int = 380,
) -> bool:
    """Render *fig* under the clip at *video_path*, cursor tracking playback.

    *fig*'s x axis must be the fraction of the episode in ``[0, 1]``, which is
    what the cursor position means: ``currentTime / duration``. That mapping is
    deliberately not routed through a frame count and an fps -- these clips are
    recorded at wall-clock speed but stepped at a few Hz, so ``frames / fps``
    disagrees with the real duration while the ratio of the two times does not.

    Returns False (having rendered nothing) when the clip cannot be served, so
    the caller can show its own empty state.
    """
    url = video_src(pathlib.Path(video_path), coordinates=f"vcp:{key}")
    if url is None:
        return False

    # A fresh id per render: two episodes' charts must never share one, or
    # Plotly.relayout drives whichever div the browser saw first.
    div_id = f"curves-{uuid.uuid4().hex[:12]}"
    fig.update_layout(shapes=[_cursor_shape()])
    chart = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )

    html = f"""
<div style="font-family: 'Source Sans Pro', sans-serif;">
  <video id="vid-{div_id}" src="{url}" controls playsinline preload="metadata"
         style="width:100%; max-height:{video_height_px}px; background:#000;"></video>
  {chart}
  <div id="warn-{div_id}" style="display:none; color:#ffb3b3; font-size:0.85rem;
       padding:0.5rem 0;">
    plotly.js could not be loaded from the CDN, so the curves are not drawn.
    The video above still plays.
  </div>
</div>
<script>
(function () {{
  const v = document.getElementById("vid-{div_id}");
  const gd = document.getElementById("{div_id}");
  if (!window.Plotly || !gd) {{
    const w = document.getElementById("warn-{div_id}");
    if (w) w.style.display = "block";
    return;
  }}

  let raf = null;
  function place() {{
    if (!v.duration || !isFinite(v.duration)) return;
    const f = Math.max(0, Math.min(1, v.currentTime / v.duration));
    Plotly.relayout(gd, {{"shapes[0].x0": f, "shapes[0].x1": f}});
  }}
  // rAF while playing: `timeupdate` fires about 4x a second, which reads as a
  // stutter against a curve the eye is tracking. Stopped when paused so a
  // parked panel costs nothing.
  function loop() {{ place(); raf = requestAnimationFrame(loop); }}
  function start() {{ if (raf === null) loop(); }}
  function stop() {{ if (raf !== null) {{ cancelAnimationFrame(raf); raf = null; }} place(); }}

  v.addEventListener("play", start);
  v.addEventListener("pause", stop);
  v.addEventListener("ended", stop);
  v.addEventListener("seeked", place);
  v.addEventListener("loadedmetadata", place);

  // Click INSIDE THE PLOT AREA to seek there. Deliberately not the whole div:
  // the legend sits in the top margin, and seeking on a legend click made
  // showing or hiding a curve jump the video -- two unrelated actions on one
  // gesture. Anything outside the axes, and anything on the legend, is left to
  // Plotly. p2d is Plotly-internal, so the whole thing is guarded: losing
  // click-to-seek is a missing nicety, not a broken panel.
  gd.addEventListener("click", function (ev) {{
    try {{
      if (ev.target.closest && ev.target.closest(".legend")) return;
      const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
      const bb = gd.getBoundingClientRect();
      const px = ev.clientX - bb.left, py = ev.clientY - bb.top;
      if (px < xa._offset || px > xa._offset + xa._length) return;
      if (py < ya._offset || py > ya._offset + ya._length) return;
      const f = xa.p2d(px - xa._offset);
      if (isFinite(f) && v.duration) {{
        v.currentTime = Math.max(0, Math.min(1, f)) * v.duration;
      }}
    }} catch (e) {{ /* no seek from this click */ }}
  }});
}})();
</script>
"""
    components.html(html, height=video_height_px + chart_height_px + 60, scrolling=False)
    return True


__all__ = ["video_curve_player", "CURSOR_COLOR"]
