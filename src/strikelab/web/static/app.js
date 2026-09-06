"use strict";

/* StrikeLab Studio front end.
   No framework, no build step: the whole app is one state object plus a render
   pass per view, which keeps it inspectable and dependency-free. */

const VERDICT_COLOR = {
  GOAL: "#3fb950",
  SAVED: "#d29922",
  WOODWORK: "#39c5cf",
  BLOCKED: "#6e7681",
  OFF_TARGET: "#f85149",
  UNRESOLVED: "#8b9bad",
};

const ZONE_NAMES = [
  "bottom-left", "bottom-centre", "bottom-right",
  "middle-left", "middle-centre", "middle-right",
  "top-left", "top-centre", "top-right",
];

const GOAL_W = 7.32;
const GOAL_H = 2.44;

const state = {
  view: "welcome",
  sessions: [],
  players: [],
  session: null,
  shots: [],
  backends: [],
  backend: "ultralytics",
  corners: [],
  groundPoints: [],
  groundMode: false,
  frameIndex: 0,
  frameImage: null,
  dragging: -1,
  events: null,
  shareToken: null,
};

const $ = (id) => document.getElementById(id);
const views = ["welcome", "upload", "calibrate", "analyse", "results", "player"];

function show(view) {
  state.view = view;
  views.forEach((name) => { $(`view-${name}`).hidden = name !== view; });
  $("sidebar").classList.remove("open");
  window.scrollTo(0, 0);
}

function toast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("err", isError);
  el.hidden = false;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.hidden = true; }, isError ? 6000 : 3000);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const isJson = (response.headers.get("content-type") || "").includes("json");
  const payload = isJson ? await response.json() : null;
  if (!response.ok) {
    throw new Error((payload && payload.detail) || `Request failed (${response.status})`);
  }
  return payload;
}

/* ── steps ──────────────────────────────────────────────── */
function renderSteps() {
  document.querySelectorAll(".steps").forEach((el) => {
    const current = Number(el.dataset.step);
    el.innerHTML = ["Upload", "Mark the goal", "Analyse"]
      .map((label, index) => {
        const n = index + 1;
        const cls = n === current ? "on" : n < current ? "done" : "";
        return `<span class="step-pill ${cls}">${n}. ${label}</span>`;
      })
      .join("");
  });
}

/* ── sidebar ────────────────────────────────────────────── */
async function loadSessions() {
  const { sessions } = await api("/api/sessions");
  state.sessions = sessions;
  $("session-count").textContent = sessions.length;
  const list = $("session-list");
  if (!sessions.length) {
    list.innerHTML = `<li class="muted small" style="cursor:default">No sessions yet</li>`;
    return;
  }
  list.innerHTML = sessions
    .map((s) => {
      const summary = s.summary || {};
      const meta = s.status === "done"
        ? `${summary.goals ?? 0}/${summary.shots ?? 0} scored`
        : s.status;
      return `<li data-id="${s.id}" class="${state.session && state.session.id === s.id ? "active" : ""}">
        <span class="status-dot ${s.status}"></span>
        <span class="s-name">${escapeHtml(s.name)}</span>
        <span class="s-meta">${escapeHtml(meta)}</span>
      </li>`;
    })
    .join("");
  list.querySelectorAll("li[data-id]").forEach((li) => {
    li.onclick = () => openSession(li.dataset.id);
  });
}

async function loadPlayers() {
  const { players } = await api("/api/players");
  state.players = players;
  const list = $("player-list");
  if (!players.length) {
    list.innerHTML = `<li class="muted small" style="cursor:default">No players yet</li>`;
    return;
  }
  list.innerHTML = players
    .map((p) => `<li data-pid="${p.id}">
      <span class="s-name">${escapeHtml(p.name)}</span>
      <span class="s-meta">${p.session_count}</span>
    </li>`)
    .join("");
  list.querySelectorAll("li[data-pid]").forEach((li) => {
    li.onclick = () => openPlayer(li.dataset.pid);
  });
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ── new session / upload ───────────────────────────────── */
function startNewSession() {
  state.session = null;
  state.shots = [];
  state.corners = [];
  state.groundPoints = [];
  $("session-name").value = "";
  $("upload-progress").hidden = true;
  $("upload-bar").style.width = "0%";
  show("upload");
}

function bindUpload() {
  const zone = $("dropzone");
  const input = $("file-input");

  $("choose-file").onclick = () => input.click();
  zone.onclick = (event) => { if (event.target === zone || event.target.closest(".dz-inner") === event.target) input.click(); };
  input.onchange = () => { if (input.files[0]) uploadFile(input.files[0]); };

  ["dragenter", "dragover"].forEach((type) =>
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add("hot"); }));
  ["dragleave", "drop"].forEach((type) =>
    zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.remove("hot"); }));
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
}

async function uploadFile(file) {
  const name = $("session-name").value.trim() || file.name.replace(/\.[^.]+$/, "");
  let session;
  try {
    session = await api("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
  } catch (error) {
    toast(error.message, true);
    return;
  }
  state.session = session;

  $("upload-progress").hidden = false;
  $("upload-label").textContent = `Uploading ${file.name}…`;

  // XHR rather than fetch: it reports real upload progress, and streams the
  // body to the server without buffering the file in memory here.
  await new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/sessions/${session.id}/video?name=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const pct = (event.loaded / event.total) * 100;
      $("upload-bar").style.width = `${pct}%`;
      $("upload-label").textContent = `Uploading ${file.name} — ${Math.round(pct)}%`;
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        state.session = JSON.parse(xhr.responseText);
        $("upload-label").textContent = "Reading the video…";
        loadSessions();
        goToCalibrate();
      } else {
        let detail = `Upload failed (${xhr.status})`;
        try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (_) {}
        toast(detail, true);
        $("upload-progress").hidden = true;
      }
      resolve();
    };
    xhr.onerror = () => { toast("Network error during upload.", true); resolve(); };
    xhr.send(file);
  });
}

/* ── calibration ────────────────────────────────────────── */
function goToCalibrate() {
  const session = state.session;
  state.corners = (session.goal_corners || []).map((p) => ({ x: p[0], y: p[1] }));
  state.groundPoints = (session.ground_points || []).map((p) => ({ x: p[0], y: p[1] }));
  state.groundMode = false;
  state.frameIndex = 0;

  const total = Math.max(0, (session.frame_count || 1) - 1);
  const slider = $("frame-slider");
  slider.max = String(total);
  slider.value = "0";
  $("frame-out").textContent = "0";

  show("calibrate");
  loadFrame(0);
}

function loadFrame(index) {
  const image = new Image();
  image.onload = () => { state.frameImage = image; drawCalib(); };
  image.onerror = () => toast("Could not read that frame.", true);
  image.src = `/api/sessions/${state.session.id}/frame?index=${index}`;
}

function drawCalib() {
  const canvas = $("calib-canvas");
  const image = state.frameImage;
  if (!image) return;

  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);

  const scale = image.naturalWidth / 900;
  const r = Math.max(5, 7 * scale);

  drawPoints(ctx, state.corners, "#35d07f", r, scale, true);
  if (state.groundPoints.length) drawPoints(ctx, state.groundPoints, "#4c8dff", r * 0.85, scale, false);

  updateCornerList();
}

function drawPoints(ctx, points, colour, radius, scale, withGrid) {
  if (points.length > 1) {
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
    if (points.length === 4) ctx.closePath();
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2.5 * scale;
    ctx.stroke();
  }

  if (withGrid && points.length === 4) {
    const ordered = orderCorners(points);
    ctx.strokeStyle = "rgba(255,255,255,.34)";
    ctx.lineWidth = 1.2 * scale;
    for (let i = 1; i < 3; i += 1) {
      const t = i / 3;
      line(ctx, lerp(ordered[0], ordered[1], t), lerp(ordered[3], ordered[2], t));
      line(ctx, lerp(ordered[0], ordered[3], t), lerp(ordered[1], ordered[2], t));
    }
  }

  points.forEach((p, index) => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.lineWidth = 2 * scale;
    ctx.strokeStyle = "#06140c";
    ctx.stroke();
    ctx.fillStyle = "#06140c";
    ctx.font = `700 ${Math.round(9 * scale)}px system-ui`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(index + 1), p.x, p.y);
  });
}

function line(ctx, a, b) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

const lerp = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });

/** Sort to top-left, top-right, bottom-right, bottom-left (image y grows down). */
function orderCorners(points) {
  const byY = [...points].sort((a, b) => a.y - b.y);
  const top = byY.slice(0, 2).sort((a, b) => a.x - b.x);
  const bottom = byY.slice(2).sort((a, b) => a.x - b.x);
  return [top[0], top[1], bottom[1], bottom[0]];
}

function updateCornerList() {
  const labels = ["Corner 1", "Corner 2", "Corner 3", "Corner 4"];
  $("corner-list").innerHTML = labels
    .map((label, index) => {
      const point = state.corners[index];
      return point
        ? `<li>${label} <span class="muted small">${Math.round(point.x)}, ${Math.round(point.y)}</span></li>`
        : `<li class="pending">${label}</li>`;
    })
    .join("");

  const groundLabels = ["Near left", "Near right", "Far right", "Far left"];
  $("ground-list").innerHTML = groundLabels
    .map((label, index) => {
      const point = state.groundPoints[index];
      return point
        ? `<li>${label} <span class="muted small">${Math.round(point.x)}, ${Math.round(point.y)}</span></li>`
        : `<li class="pending">${label}</li>`;
    })
    .join("");

  const ready = state.corners.length === 4;
  $("calib-continue").disabled = !ready;
  $("calib-hint").textContent = state.groundMode
    ? state.groundPoints.length < 4
      ? `Ground point ${state.groundPoints.length + 1} of 4.`
      : "Ground reference set."
    : ready
      ? "All four corners set. Drag to adjust, or continue."
      : `Click corner ${state.corners.length + 1} of 4.`;
}

function canvasPoint(event) {
  const canvas = $("calib-canvas");
  const rect = canvas.getBoundingClientRect();
  const source = event.touches ? event.touches[0] : event;
  return {
    x: ((source.clientX - rect.left) / rect.width) * canvas.width,
    y: ((source.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function bindCalibration() {
  const canvas = $("calib-canvas");

  const hitTest = (point) => {
    const list = state.groundMode ? state.groundPoints : state.corners;
    const tolerance = canvas.width / 45;
    for (let i = 0; i < list.length; i += 1) {
      if (Math.hypot(list[i].x - point.x, list[i].y - point.y) < tolerance) return i;
    }
    return -1;
  };

  const down = (event) => {
    event.preventDefault();
    const point = canvasPoint(event);
    const hit = hitTest(point);
    if (hit >= 0) { state.dragging = hit; return; }
    const list = state.groundMode ? state.groundPoints : state.corners;
    if (list.length < 4) { list.push(point); drawCalib(); }
  };

  const move = (event) => {
    if (state.dragging < 0) return;
    event.preventDefault();
    const list = state.groundMode ? state.groundPoints : state.corners;
    list[state.dragging] = canvasPoint(event);
    drawCalib();
  };

  const up = () => { state.dragging = -1; };

  canvas.addEventListener("pointerdown", down);
  canvas.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);

  $("frame-slider").oninput = (event) => {
    const index = Number(event.target.value);
    $("frame-out").textContent = String(index);
    state.frameIndex = index;
    loadFrame(index);
  };

  $("reset-corners").onclick = () => {
    if (state.groundMode) state.groundPoints = [];
    else state.corners = [];
    drawCalib();
  };

  $("ground-toggle").onclick = () => {
    state.groundMode = !state.groundMode;
    $("ground-list").hidden = !state.groundMode;
    $("ground-toggle").textContent = state.groundMode
      ? "Done marking ground"
      : "Mark 4 ground points";
    updateCornerList();
  };

  $("calib-continue").onclick = saveCalibration;
}

async function saveCalibration() {
  const body = {
    goal_corners: state.corners.map((p) => [p.x, p.y]),
  };
  if (state.groundPoints.length === 4) {
    body.ground_points = state.groundPoints.map((p) => [p.x, p.y]);
    body.ground_preset = $("ground-preset").value;
  }
  try {
    state.session = await api(`/api/sessions/${state.session.id}/calibration`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    goToAnalyse();
  } catch (error) {
    toast(error.message, true);
  }
}

/* ── analyse ────────────────────────────────────────────── */
async function loadBackends() {
  const { backends } = await api("/api/backends");
  state.backends = backends;
  const usable = backends.find((b) => b.available);
  if (usable) state.backend = usable.name;

  $("backend-options").innerHTML = backends
    .map((b) => `
      <label class="radio-opt ${b.available ? "" : "disabled"} ${b.name === state.backend ? "sel" : ""}" data-b="${b.name}">
        <input type="radio" name="backend" value="${b.name}" ${b.name === state.backend ? "checked" : ""} ${b.available ? "" : "disabled"} />
        <span>
          <span class="r-name">${b.name}</span>
          <span class="r-desc">${escapeHtml(b.available ? b.description : b.reason)}</span>
        </span>
      </label>`)
    .join("");

  $("backend-options").querySelectorAll(".radio-opt").forEach((el) => {
    el.onclick = () => {
      const backend = state.backends.find((b) => b.name === el.dataset.b);
      if (!backend || !backend.available) return;
      state.backend = el.dataset.b;
      $("backend-options").querySelectorAll(".radio-opt").forEach((o) => o.classList.remove("sel"));
      el.classList.add("sel");
    };
  });

  const note = backends.filter((b) => !b.available).map((b) => `${b.name}: ${b.reason}`);
  $("backend-note").textContent = note.length ? note.join(" · ") : "";
}

function goToAnalyse() {
  $("analysis-progress").hidden = true;
  show("analyse");
}

function bindAnalyse() {
  const sliders = [
    ["confidence", "conf-out", (v) => v],
    ["release-speed", "rs-out", (v) => Number(v).toFixed(1)],
    ["release-accel", "ra-out", (v) => Number(v).toFixed(1)],
    ["contact-radius", "cr-out", (v) => Number(v).toFixed(1)],
  ];
  sliders.forEach(([id, out, fmt]) => {
    $(id).oninput = () => { $(out).textContent = fmt($(id).value); };
  });

  $("reset-thresholds").onclick = () => {
    $("release-speed").value = 7; $("rs-out").textContent = "7.0";
    $("release-accel").value = 1.8; $("ra-out").textContent = "1.8";
    $("contact-radius").value = 2.6; $("cr-out").textContent = "2.6";
  };

  $("start-analysis").onclick = startAnalysis;
  $("cancel-analysis").onclick = async () => {
    try { await api(`/api/sessions/${state.session.id}/cancel`, { method: "POST" }); }
    catch (error) { toast(error.message, true); }
  };
}

async function startAnalysis() {
  const body = {
    backend: state.backend,
    confidence: Number($("confidence").value),
    write_video: $("write-video").checked,
    thresholds: {
      release_speed_mps: Number($("release-speed").value),
      release_accel_ratio: Number($("release-accel").value),
      contact_radius_ball_diameters: Number($("contact-radius").value),
    },
  };
  try {
    await api(`/api/sessions/${state.session.id}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    toast(error.message, true);
    return;
  }
  $("analysis-progress").hidden = false;
  $("analysis-bar").style.width = "0%";
  $("analysis-label").textContent = "Starting…";
  listenToJob();
  loadSessions();
}

function listenToJob() {
  if (state.events) state.events.close();
  const source = new EventSource(`/api/sessions/${state.session.id}/events`);
  state.events = source;

  source.addEventListener("progress", (event) => {
    const data = JSON.parse(event.data);
    const pct = (data.progress || 0) * 100;
    $("analysis-bar").style.width = `${pct}%`;
    const found = data.shots_found ? ` · ${data.shots_found} shot${data.shots_found === 1 ? "" : "s"}` : "";
    $("analysis-label").textContent = `${data.message || "Working"}${found}`;
  });

  const finish = async (event, ok) => {
    source.close();
    state.events = null;
    if (ok) {
      toast("Analysis complete.");
      await openSession(state.session.id);
    } else {
      const data = event.data ? JSON.parse(event.data) : {};
      toast(data.error || "Analysis failed.", true);
      $("analysis-label").textContent = data.error || "Failed";
    }
    loadSessions();
  };

  source.addEventListener("done", (event) => finish(event, true));
  source.addEventListener("failed", (event) => finish(event, false));
  source.addEventListener("cancelled", () => {
    source.close();
    state.events = null;
    toast("Analysis cancelled.");
    $("analysis-label").textContent = "Cancelled";
    loadSessions();
  });
  source.onerror = () => { /* EventSource retries on its own */ };
}

/* ── results ────────────────────────────────────────────── */
async function openSession(sessionId) {
  let session;
  try { session = await api(`/api/sessions/${sessionId}`); }
  catch (error) { toast(error.message, true); return; }

  state.session = session;
  state.shots = session.shots || [];
  loadSessions();

  if (session.status === "analysing" && session.job) {
    goToAnalyse();
    $("analysis-progress").hidden = false;
    listenToJob();
    return;
  }
  if (!session.video_path) { startNewSession(); return; }
  if (!session.goal_corners) { goToCalibrate(); return; }
  if (session.status !== "done" && session.status !== "cancelled" && !state.shots.length) {
    goToAnalyse();
    return;
  }
  renderResults();
}

function renderResults() {
  const session = state.session;
  const summary = session.summary || {};
  const shots = state.shots;

  $("results-title").textContent = session.name;
  const when = session.created_at ? new Date(session.created_at).toLocaleString() : "";
  const player = state.players.find((p) => p.id === session.player_id);
  $("results-sub").textContent =
    `${when}${player ? ` · ${player.name}` : ""} · ${session.backend || "?"} · ${session.frame_count || 0} frames`;

  renderResultActions();
  renderTiles($("tiles"), summary, shots);
  renderGoalMap($("goalmap"), shots);
  renderLegend($("goalmap-legend"), shots);
  renderZoneGrid($("zonegrid"), shots);

  const card = $("video-card");
  if (session.annotated_path) {
    card.hidden = false;
    $("result-video").src = `/api/sessions/${session.id}/annotated`;
  } else {
    card.hidden = true;
    $("result-video").removeAttribute("src");
  }

  renderShotTable(shots);
  show("results");
}

function renderResultActions() {
  const session = state.session;
  const options = state.players
    .map((p) => `<option value="${p.id}" ${p.id === session.player_id ? "selected" : ""}>${escapeHtml(p.name)}</option>`)
    .join("");

  $("results-actions").innerHTML = `
    <select id="assign-player" style="width:auto">
      <option value="">No player</option>${options}
    </select>
    <a class="btn btn-sm" href="/api/sessions/${session.id}/export?format=csv">CSV</a>
    <a class="btn btn-sm" href="/api/sessions/${session.id}/export?format=json">JSON</a>
    <button class="btn btn-sm" id="share-btn">${session.share_token ? "Copy link" : "Share"}</button>
    <button class="btn btn-sm btn-danger" id="delete-btn">Delete</button>`;

  $("assign-player").onchange = async (event) => {
    try {
      state.session = await api(`/api/sessions/${state.session.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_id: event.target.value || null }),
      });
      toast("Player updated.");
      loadPlayers();
      loadSessions();
    } catch (error) { toast(error.message, true); }
  };

  $("share-btn").onclick = async () => {
    try {
      const result = await api(`/api/sessions/${state.session.id}/share`, { method: "POST" });
      const url = `${location.origin}${result.url}`;
      state.session.share_token = result.share_token;
      try { await navigator.clipboard.writeText(url); toast("Share link copied."); }
      catch (_) { toast(url); }
      renderResultActions();
    } catch (error) { toast(error.message, true); }
  };

  $("delete-btn").onclick = async () => {
    if (!confirm(`Delete "${state.session.name}" and its video?`)) return;
    try {
      await api(`/api/sessions/${state.session.id}`, { method: "DELETE" });
      state.session = null;
      await loadSessions();
      show("welcome");
      toast("Session deleted.");
    } catch (error) { toast(error.message, true); }
  };
}

function renderTiles(container, summary, shots) {
  const speeds = shots.map((s) => s.peak_speed_kmh).filter(Boolean);
  const best = speeds.length ? Math.max(...speeds) : 0;
  const tiles = [
    ["shots", summary.shots ?? shots.length],
    ["goals", summary.goals ?? 0],
    ["on target", `${Math.round((summary.on_target_rate ?? 0) * 100)}%`],
    ["conversion", `${Math.round((summary.conversion_rate ?? 0) * 100)}%`],
    ["best speed", best ? `${Math.round(best)} km/h` : "—"],
    ["avg speed", summary.mean_speed_kmh ? `${Math.round(summary.mean_speed_kmh)} km/h` : "—"],
  ];
  container.innerHTML = tiles
    .map(([k, v]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`)
    .join("");
}

/** The money view: the goal mouth seen from the pitch, with every shot on it. */
function renderGoalMap(container, shots) {
  const pad = 0.9;
  const W = GOAL_W + pad * 2;
  const H = GOAL_H + pad * 2;
  const sx = (x) => ((x + pad) / W) * 100;
  const sy = (y) => ((GOAL_H - y + pad) / H) * 100;

  const grid = [];
  for (let i = 1; i < 3; i += 1) {
    grid.push(`<line x1="${sx(GOAL_W * i / 3)}" y1="${sy(GOAL_H)}" x2="${sx(GOAL_W * i / 3)}" y2="${sy(0)}" stroke="#26313d" stroke-width="0.25"/>`);
    grid.push(`<line x1="${sx(0)}" y1="${sy(GOAL_H * i / 3)}" x2="${sx(GOAL_W)}" y2="${sy(GOAL_H * i / 3)}" stroke="#26313d" stroke-width="0.25"/>`);
  }

  const dots = shots.map((shot) => {
    const point = (shot.entry || {}).goal_plane_m;
    const colour = VERDICT_COLOR[shot.verdict] || "#8b9bad";
    if (!point) return "";
    const cx = sx(point[0]);
    const cy = sy(point[1]);
    const inside = point[0] >= 0 && point[0] <= GOAL_W && point[1] >= 0 && point[1] <= GOAL_H;
    return `<circle cx="${cx}" cy="${cy}" r="${inside ? 1.5 : 1.2}" fill="${colour}"
      fill-opacity="${inside ? 0.9 : 0.55}" stroke="#0b0f14" stroke-width="0.3">
      <title>#${shot.shot_id} ${shot.verdict} — ${Math.round(shot.peak_speed_kmh || 0)} km/h${shot.zone ? ` (${shot.zone})` : ""}</title>
    </circle>`;
  }).join("");

  container.innerHTML = `
    <svg viewBox="0 0 100 ${(H / W) * 100}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Goal mouth placement">
      <rect x="0" y="0" width="100" height="${(H / W) * 100}" fill="#0b0f14" rx="1"/>
      ${grid.join("")}
      <rect x="${sx(0)}" y="${sy(GOAL_H)}" width="${sx(GOAL_W) - sx(0)}" height="${sy(0) - sy(GOAL_H)}"
            fill="none" stroke="#e8eef5" stroke-width="0.7"/>
      <line x1="${sx(-pad)}" y1="${sy(0)}" x2="${sx(GOAL_W + pad)}" y2="${sy(0)}" stroke="#26313d" stroke-width="0.4"/>
      ${dots}
    </svg>`;
}

function renderLegend(container, shots) {
  const counts = {};
  shots.forEach((s) => { counts[s.verdict] = (counts[s.verdict] || 0) + 1; });
  container.innerHTML = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([verdict, count]) =>
      `<span><i style="background:${VERDICT_COLOR[verdict] || "#8b9bad"}"></i>${verdict.toLowerCase().replace("_", " ")} ${count}</span>`)
    .join("") || `<span class="muted">No shots</span>`;
}

function renderZoneGrid(container, shots) {
  const counts = {};
  shots.forEach((s) => { if (s.zone) counts[s.zone] = (counts[s.zone] || 0) + 1; });
  const max = Math.max(1, ...Object.values(counts));

  // Rows top to bottom for display: top, middle, bottom.
  const rows = [6, 3, 0];
  const cells = rows.flatMap((rowStart, rowIndex) =>
    [0, 1, 2].map((col) => {
      const zone = ZONE_NAMES[rowStart + col];
      const count = counts[zone] || 0;
      const alpha = count ? 0.15 + 0.75 * (count / max) : 0;
      return `<g>
        <rect x="${col * 33.3}" y="${rowIndex * 22}" width="32.3" height="21" rx="1.5"
              fill="#35d07f" fill-opacity="${alpha}" stroke="#26313d" stroke-width="0.3"/>
        <text x="${col * 33.3 + 16}" y="${rowIndex * 22 + 12}" text-anchor="middle"
              font-size="8" font-weight="700" fill="${count ? "#e8eef5" : "#3a4553"}">${count}</text>
      </g>`;
    }));

  container.innerHTML = `<svg viewBox="0 0 100 66" role="img" aria-label="Shots by zone">${cells.join("")}</svg>`;
}

function renderShotTable(shots) {
  const body = document.querySelector("#shot-table tbody");
  if (!shots.length) {
    body.innerHTML = `<tr><td colspan="8" class="empty">No shots were detected. Try lowering the minimum shot speed and re-running.</td></tr>`;
    return;
  }
  body.innerHTML = shots.map((shot) => {
    const colour = VERDICT_COLOR[shot.verdict] || "#8b9bad";
    const quality = (shot.quality || {}).score;
    return `<tr>
      <td class="num">${shot.shot_id}</td>
      <td><span class="pill" style="background:${colour}22;color:${colour}">${shot.verdict}</span></td>
      <td class="num">${shot.peak_speed_kmh ? `${Math.round(shot.peak_speed_kmh)} km/h` : "—"}</td>
      <td>${shot.zone || `<span class="muted">off target</span>`}</td>
      <td class="num">${quality != null ? quality.toFixed(3) : "—"}</td>
      <td class="num">${shot.distance_m != null ? `${shot.distance_m.toFixed(1)} m` : "—"}</td>
      <td class="muted">${shot.confidence || ""}</td>
      <td class="note-cell">${escapeHtml((shot.notes || [])[0] || "")}</td>
    </tr>`;
  }).join("");
}

/* ── players ────────────────────────────────────────────── */
async function openPlayer(playerId) {
  let data;
  try { data = await api(`/api/players/${playerId}/shots`); }
  catch (error) { toast(error.message, true); return; }

  const { player, shots } = data;
  $("player-title").textContent = player.name;

  const sessions = [...new Set(shots.map((s) => s.session_id))];
  const goals = shots.filter((s) => s.verdict === "GOAL").length;
  const onTarget = shots.filter((s) => s.on_target).length;
  $("player-sub").textContent = `${sessions.length} session${sessions.length === 1 ? "" : "s"} · ${shots.length} shots`;

  renderTiles($("player-tiles"), {
    shots: shots.length,
    goals,
    on_target_rate: shots.length ? onTarget / shots.length : 0,
    conversion_rate: shots.length ? goals / shots.length : 0,
    mean_speed_kmh: shots.length
      ? shots.reduce((sum, s) => sum + (s.peak_speed_kmh || 0), 0) / shots.length
      : 0,
  }, shots);

  renderGoalMap($("player-goalmap"), shots);
  renderTrend($("player-trend"), shots);
  show("player");
}

function renderTrend(container, shots) {
  const bySession = new Map();
  shots.forEach((shot) => {
    const key = shot.session_id;
    if (!bySession.has(key)) bySession.set(key, { name: shot.session_name, total: 0, goals: 0 });
    const entry = bySession.get(key);
    entry.total += 1;
    if (shot.verdict === "GOAL") entry.goals += 1;
  });

  const entries = [...bySession.values()];
  if (!entries.length) { container.innerHTML = `<p class="muted small">No sessions yet.</p>`; return; }

  const width = 100;
  const height = 46;
  const barWidth = Math.min(14, (width - 4) / entries.length - 2);
  const bars = entries.map((entry, index) => {
    const rate = entry.total ? entry.goals / entry.total : 0;
    const h = Math.max(1, rate * (height - 12));
    const x = 2 + index * ((width - 4) / entries.length);
    return `<g>
      <rect x="${x}" y="${height - 8 - h}" width="${barWidth}" height="${h}" rx="1" fill="#35d07f"/>
      <text x="${x + barWidth / 2}" y="${height - 1}" font-size="3.4" fill="#8b9bad" text-anchor="middle">${Math.round(rate * 100)}%</text>
      <title>${escapeHtml(entry.name)}: ${entry.goals}/${entry.total}</title>
    </g>`;
  }).join("");

  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Conversion by session">${bars}</svg>`;
}

/* ── shared view ────────────────────────────────────────── */
async function loadSharedIfNeeded() {
  const match = location.pathname.match(/^\/s\/(.+)$/);
  if (!match) return false;
  try {
    const session = await api(`/api/shared/${match[1]}`);
    state.session = session;
    state.shots = session.shots || [];
    document.querySelector(".sidebar").hidden = true;
    $("new-session-btn").hidden = true;
    $("burger").hidden = true;
    document.querySelector(".shell").style.gridTemplateColumns = "1fr";
    renderResults();
    $("results-actions").innerHTML = `<span class="muted small">Shared session (read only)</span>`;
    $("video-card").hidden = true;
    return true;
  } catch (error) {
    toast(error.message, true);
    show("welcome");
    return true;
  }
}

/* ── boot ───────────────────────────────────────────────── */
async function boot() {
  renderSteps();
  bindUpload();
  bindCalibration();
  bindAnalyse();

  $("new-session-btn").onclick = startNewSession;
  $("welcome-start").onclick = startNewSession;
  $("burger").onclick = () => $("sidebar").classList.toggle("open");

  $("add-player-btn").onclick = async () => {
    const name = prompt("Player name");
    if (!name || !name.trim()) return;
    try {
      await api("/api/players", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim() }),
      });
      loadPlayers();
    } catch (error) { toast(error.message, true); }
  };

  if (await loadSharedIfNeeded()) return;

  await Promise.all([loadSessions(), loadPlayers(), loadBackends()]);
  if (state.sessions.length) openSession(state.sessions[0].id);
  else show("welcome");
}

boot();
