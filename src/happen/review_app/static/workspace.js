(() => {
  "use strict";

  function $(id) {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Missing element #${id}`);
    return el;
  }

  function readInjectedJSON(id) {
    const el = $(id);
    const txt = el.textContent || "";
    return JSON.parse(txt);
  }

  function safeStr(x) {
    return x === null || x === undefined ? "" : String(x);
  }

  function normSession(s) {
    return safeStr(s).trim();
  }

  function normScanUid(s) {
    return safeStr(s).trim();
  }

  function fmtSession(s) {
    const t = normSession(s);
    return t ? t : "(no session)";
  }

  function fmtSimilarity(sim) {
    const v = Number(sim);
    if (!Number.isFinite(v)) return "N/A";
    return v.toFixed(6);
  }

  function getQueryParam(name) {
    return new URLSearchParams(window.location.search).get(name);
  }

  function clamp(x, lo, hi) {
    return x < lo ? lo : x > hi ? hi : x;
  }

  function setHidden(el, hidden) {
    el.classList.toggle("is-hidden", !!hidden);
  }

  function buildPngUrl(scanObj) {
    const ds = safeStr(scanObj.dataset).trim();
    const sbj = safeStr(scanObj.subject_id).trim();
    const ses = normSession(scanObj.session_id);
    const uid = normScanUid(scanObj.scan_uid);

    if (!ds || !sbj || !uid) {
      throw new Error("Missing dataset/subject_id/scan_uid in scanObj");
    }

    const params = new URLSearchParams();
    params.set("dataset", ds);
    params.set("subject_id", sbj);
    if (ses) params.set("session_id", ses);
    params.set("scan_uid", uid);

    return `/png?${params.toString()}`;
  }

  function buildPairAssetUrl(kind, scanUidA, scanUidB) {
    const a = normScanUid(scanUidA);
    const b = normScanUid(scanUidB);
    if (!a || !b) {
      throw new Error("Missing scan_uid_a/scan_uid_b for pair asset");
    }

    const u = new URL(PAIR_ASSET_URL, window.location.origin);
    u.searchParams.set("kind", safeStr(kind).trim().toLowerCase());
    u.searchParams.set("scan_uid_a", a);
    u.searchParams.set("scan_uid_b", b);

    return (
      u.pathname +
      (u.searchParams.toString() ? `?${u.searchParams.toString()}` : "")
    );
  }

  function getPairKey(cand) {
    const key = safeStr(cand && cand.pair_key).trim();
    if (!key) {
      throw new Error("Missing candidate.pair_key in workspace payload");
    }
    return key;
  }

  function metaText(scanObj, isCandidate) {
    const ds = safeStr(scanObj.dataset).trim();
    const sbj = safeStr(scanObj.subject_id).trim();
    const ses = fmtSession(scanObj.session_id);

    const base = `Dataset: ${ds}\nSubject: ${sbj}\nSession: ${ses}\n`;
    if (isCandidate) {
      return base + `Similarity: ${fmtSimilarity(scanObj.similarity)}`;
    }
    return base + "\u00A0";
  }

  async function postJSON(url, payload) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const txt = await resp.text().catch(() => "");
      throw new Error(
        `POST ${url} failed: ${resp.status} ${resp.statusText} ${txt}`,
      );
    }
    return await resp.json();
  }

  function setSelectValueStrict(selectEl, value) {
    for (const opt of selectEl.options) {
      if (opt.value === value) {
        selectEl.value = value;
        return;
      }
    }
    throw new Error(
      `Value ${value} not found in <select id="${selectEl.id}"> options`,
    );
  }

  function decisionToPlain(entry) {
    if (!entry) return null;
    return {
      qa_status: entry.qa_status,
      reason: entry.reason,
      date: entry.serverDate || "",
    };
  }

  const BOOTSTRAP = readInjectedJSON("workspace-bootstrap");
  const WORKSPACE_API_URL = readInjectedJSON("workspace-api-url");
  const PAIR_ASSET_URL = readInjectedJSON("pair-asset-url");
  const DATASETS_INDEX_URL = readInjectedJSON("datasets-index-url");

  const loadingView = $("loadingView");
  const subjectsView = $("subjectsView");
  const reviewView = $("reviewView");

  const loadingStageText = $("loadingStageText");
  const loadingBarFill = $("loadingBarFill");
  const loadingProgressText = $("loadingProgressText");

  const subjectsDataset = $("subjectsDataset");
  const subjectsProgressText = $("subjectsProgressText");
  const subjectsList = $("subjectsList");
  const btnBackToDatasets = $("btnBackToDatasets");

  const reviewDataset = $("reviewDataset");
  const reviewSubjectId = $("reviewSubjectId");
  const btnBackToSubjects = $("btnBackToSubjects");

  const querySel = $("querySelect");
  const queryMeta = $("queryMeta");
  const candMeta = $("candMeta");

  const queryImg = $("queryImg");
  const candImg = $("candImg");
  const queryMissing = $("queryImgMissing");
  const candMissing = $("candImgMissing");

  const btnYes = $("btnYes");
  const btnNo = $("btnNo");
  const btnMaybe = $("btnMaybe");
  const reasonBox = $("reasonBox");

  const candCounter = $("candCounter");
  const diffCounter = $("diffCounter");
  const checkCounter = $("checkCounter");

  const diffCanvas = $("diffCanvas");
  const diffMissing = $("diffMissing");
  const diffCtx = diffCanvas.getContext("2d", { willReadFrequently: true });
  if (!diffCtx) throw new Error("diffCanvas.getContext returned null");

  const checkCanvas = $("checkCanvas");
  const checkMissing = $("checkMissing");
  const checkCtx = checkCanvas.getContext("2d", { willReadFrequently: true });
  if (!checkCtx) throw new Error("checkCanvas.getContext returned null");

  const overlay = document.getElementById("globalLoadingOverlay");
  try {
    sessionStorage.removeItem("navLoading");
  } catch (e) {}
  document.documentElement.classList.remove("nav-loading");
  if (overlay) overlay.classList.add("is-hidden");

  const DIFF_MAX = 64;
  const CHECK_TILE = 32;
  const FLUSH_DEBOUNCE_MS = 300;

  const state = {
    dataset: safeStr(BOOTSTRAP.dataset).trim(),
    reviewMode: safeStr(BOOTSTRAP.review_mode).trim().toLowerCase(),

    subjects: [],
    reviewBySubject: {},
    datasetDecisionMap: {},
    doneSubjects: 0,
    totalSubjects: 0,

    currentView: "loading",
    currentSubjectId: "",
    queryIdx: 0,
    candIdx: 0,
    subjectsScrollY: 0,

    renderVersion: 0,

    imageLoadCache: new Map(),
    lazyPairAssetCache: new Map(),
    decisionCache: new Map(),

    flushTimer: null,
    flushInFlight: false,
  };

  if (!state.dataset) {
    throw new Error("workspace bootstrap is missing dataset");
  }
  if (!["lazy", "precompute"].includes(state.reviewMode)) {
    throw new Error(`Invalid review_mode: ${state.reviewMode}`);
  }

  function showView(viewName) {
    state.currentView = viewName;
    setHidden(loadingView, viewName !== "loading");
    setHidden(subjectsView, viewName !== "subjects");
    setHidden(reviewView, viewName !== "review");
  }

  function updateTitle() {
    if (state.currentView === "subjects") {
      document.title = `Subjects | ${state.dataset}`;
      return;
    }
    if (state.currentView === "review") {
      document.title = `Review | ${state.dataset} / ${state.currentSubjectId}`;
      return;
    }
    document.title = `Workspace | ${state.dataset}`;
  }

  function setLoadingStage(text) {
    loadingStageText.textContent = text;
  }

  function setLoadingProgress(done, total) {
    const pct = total > 0 ? Math.round((done / total) * 100) : 0;
    loadingBarFill.style.width = `${pct}%`;
    loadingProgressText.textContent = `${done} / ${total}`;
  }

  function saveSubjectsScroll() {
    state.subjectsScrollY = window.scrollY || 0;
  }

  function setSelected(status) {
    const s = safeStr(status).toLowerCase();
    btnYes.classList.toggle("is-selected", s === "yes");
    btnNo.classList.toggle("is-selected", s === "no");
    btnMaybe.classList.toggle("is-selected", s === "maybe");
  }

  function currentSelectedStatus() {
    if (btnYes.classList.contains("is-selected")) return "yes";
    if (btnMaybe.classList.contains("is-selected")) return "maybe";
    return "no";
  }

  function applyDecisionToUI(decisionOrNull) {
    if (decisionOrNull && decisionOrNull.qa_status) {
      setSelected(decisionOrNull.qa_status);
      reasonBox.value =
        typeof decisionOrNull.reason === "string" ? decisionOrNull.reason : "";
    } else {
      setSelected("no");
      reasonBox.value = "";
    }
  }

  function subjectIds() {
    return state.subjects.map((r) => safeStr(r.subject_id));
  }

  function currentSubjectPayload() {
    const payload = state.reviewBySubject[state.currentSubjectId];
    if (!payload) {
      throw new Error(
        `Missing review payload for subject: ${state.currentSubjectId}`,
      );
    }
    return payload;
  }

  function currentSubjectData() {
    return currentSubjectPayload().subject_data || [];
  }

  function currentQueryBlock() {
    const data = currentSubjectData();
    const qb = data[state.queryIdx];
    if (!qb) throw new Error(`queryIdx out of range: ${state.queryIdx}`);
    return qb;
  }

  function currentQuery() {
    return currentQueryBlock().query;
  }

  function currentCandidates() {
    return currentQueryBlock().candidates || [];
  }

  function currentCandidate() {
    const cands = currentCandidates();
    if (cands.length === 0) return null;
    const i = ((state.candIdx % cands.length) + cands.length) % cands.length;
    return cands[i];
  }

  function currentSubjectNav() {
    const ids = subjectIds();
    const pos = ids.indexOf(state.currentSubjectId);
    if (pos < 0) {
      return { prev: null, next: null };
    }
    return {
      prev: pos > 0 ? ids[pos - 1] : null,
      next: pos < ids.length - 1 ? ids[pos + 1] : null,
    };
  }

  function getOrCreateDecisionEntry(q, c) {
    const key = getPairKey(c);
    let entry = state.decisionCache.get(key);

    if (!entry) {
      const seeded = state.datasetDecisionMap[key] || null;
      entry = {
        key,
        a: q,
        b: c,
        qa_status: seeded
          ? safeStr(seeded.qa_status).toLowerCase() || "no"
          : "no",
        reason:
          seeded && typeof seeded.reason === "string" ? seeded.reason : "",
        serverDate: seeded ? safeStr(seeded.date) : "",
        dirty: false,
        inflightSave: false,
        localRevision: 0,
      };
      state.decisionCache.set(key, entry);
    } else {
      entry.a = q;
      entry.b = c;
    }

    return entry;
  }

  function seedDecisionCacheFromPayload() {
    for (const subjectId of Object.keys(state.reviewBySubject)) {
      const payload = state.reviewBySubject[subjectId];
      const subjectData = Array.isArray(payload.subject_data)
        ? payload.subject_data
        : [];

      for (const qb of subjectData) {
        const q = qb.query;
        const cands = Array.isArray(qb.candidates) ? qb.candidates : [];

        for (const c of cands) {
          const key = getPairKey(c);
          const dj = state.datasetDecisionMap[key] || null;

          state.decisionCache.set(key, {
            key,
            a: q,
            b: c,
            qa_status: dj ? safeStr(dj.qa_status).toLowerCase() || "no" : "no",
            reason: dj && typeof dj.reason === "string" ? dj.reason : "",
            serverDate: dj ? safeStr(dj.date) : "",
            dirty: false,
            inflightSave: false,
            localRevision: 0,
          });
        }
      }
    }
  }

  function currentPairInfo() {
    const q = currentQuery();
    const c = currentCandidate();
    if (!c) return null;
    const entry = getOrCreateDecisionEntry(q, c);
    return { q, c, key: entry.key, entry };
  }

  function captureCurrentPairFromUI() {
    if (state.currentView !== "review") return;

    const info = currentPairInfo();
    if (!info) return;

    const status = currentSelectedStatus();
    const reason = reasonBox.value || "";

    if (info.entry.qa_status === status && info.entry.reason === reason) {
      return;
    }

    info.entry.qa_status = status;
    info.entry.reason = reason;
    info.entry.dirty = true;
    info.entry.localRevision += 1;

    scheduleFlushDirtyPairs();
  }

  function setDecisionLocal(status) {
    const info = currentPairInfo();
    if (!info) return;

    const nextStatus = safeStr(status).toLowerCase();
    if (!["yes", "no", "maybe"].includes(nextStatus)) return;

    setSelected(nextStatus);

    if (
      info.entry.qa_status === nextStatus &&
      info.entry.reason === (reasonBox.value || "")
    ) {
      return;
    }

    info.entry.qa_status = nextStatus;
    info.entry.reason = reasonBox.value || "";
    info.entry.dirty = true;
    info.entry.localRevision += 1;

    scheduleFlushDirtyPairs();
  }

  async function flushDirtyPairs() {
    if (state.flushInFlight) return;
    state.flushInFlight = true;

    try {
      while (true) {
        const dirtyEntries = Array.from(state.decisionCache.values()).filter(
          (entry) => entry.dirty && !entry.inflightSave,
        );

        if (dirtyEntries.length === 0) break;

        const revisions = new Map();
        for (const entry of dirtyEntries) {
          entry.inflightSave = true;
          revisions.set(entry.key, entry.localRevision);
        }

        const payload = {
          updates: dirtyEntries.map((entry) => ({
            a: entry.a,
            b: entry.b,
            qa_status: entry.qa_status,
            reason: entry.reason,
          })),
        };

        try {
          const res = await postJSON("/api/flush_decisions", payload);
          const decisionMap =
            res && typeof res === "object" && res.decision_map
              ? res.decision_map
              : {};

          for (const entry of dirtyEntries) {
            const rev = revisions.get(entry.key);
            if (entry.localRevision === rev) {
              entry.dirty = false;

              const dj = decisionMap[entry.key];
              if (dj) {
                entry.qa_status =
                  safeStr(dj.qa_status).toLowerCase() || entry.qa_status;
                entry.reason =
                  typeof dj.reason === "string" ? dj.reason : entry.reason;
                entry.serverDate = safeStr(dj.date);
              }
            }
          }
        } catch (err) {
          console.warn("[workspace] flushDirtyPairs failed:", err);
        } finally {
          for (const entry of dirtyEntries) {
            entry.inflightSave = false;
          }
        }
      }
    } finally {
      state.flushInFlight = false;
    }
  }

  function scheduleFlushDirtyPairs(delay = FLUSH_DEBOUNCE_MS) {
    if (state.flushTimer !== null) {
      clearTimeout(state.flushTimer);
    }

    state.flushTimer = window.setTimeout(() => {
      state.flushTimer = null;
      void flushDirtyPairs();
    }, delay);
  }

  async function flushAllDirtyPairs() {
    if (state.flushTimer !== null) {
      clearTimeout(state.flushTimer);
      state.flushTimer = null;
    }
    await flushDirtyPairs();
  }

  async function fetchWorkspaceDataset() {
    const u = new URL(WORKSPACE_API_URL, window.location.origin);
    u.searchParams.set("dataset", state.dataset);
    u.searchParams.set("include_subject_data", "1");

    const resp = await fetch(u.toString(), { method: "GET" });
    if (!resp.ok) {
      const txt = await resp.text().catch(() => "");
      throw new Error(
        `GET ${u.toString()} failed: ${resp.status} ${resp.statusText} ${txt}`,
      );
    }

    const data = await resp.json();
    if (!data.ok) {
      throw new Error("workspace dataset api returned not ok");
    }

    state.reviewMode = safeStr(data.review_mode).trim().toLowerCase();
    state.subjects = Array.isArray(data.subjects) ? data.subjects : [];
    state.reviewBySubject = data.review_by_subject || {};
    state.datasetDecisionMap =
      data && typeof data === "object" && data.decision_map
        ? data.decision_map
        : {};
    state.doneSubjects = Number(data.done_subjects) || 0;
    state.totalSubjects = Number(data.total_subjects) || 0;

    state.decisionCache.clear();
    seedDecisionCacheFromPayload();
  }

  function collectPngUrls() {
    const urls = new Set();

    for (const subjectId of Object.keys(state.reviewBySubject)) {
      const payload = state.reviewBySubject[subjectId];
      const subjectData = Array.isArray(payload.subject_data)
        ? payload.subject_data
        : [];

      for (const qb of subjectData) {
        if (qb.query) {
          urls.add(buildPngUrl(qb.query));
        }
        const cands = Array.isArray(qb.candidates) ? qb.candidates : [];
        for (const c of cands) {
          urls.add(buildPngUrl(c));
        }
      }
    }

    return Array.from(urls);
  }

  function collectAllPairs() {
    const pairs = [];
    const seen = new Set();

    for (const subjectId of Object.keys(state.reviewBySubject)) {
      const payload = state.reviewBySubject[subjectId];
      const subjectData = Array.isArray(payload.subject_data)
        ? payload.subject_data
        : [];

      for (const qb of subjectData) {
        const q = qb.query;
        if (!q) continue;

        const qUrl = buildPngUrl(q);
        const cands = Array.isArray(qb.candidates) ? qb.candidates : [];

        for (const c of cands) {
          const key = getPairKey(c);
          if (seen.has(key)) continue;
          seen.add(key);

          pairs.push({
            key,
            q,
            c,
            qUrl,
            cUrl: buildPngUrl(c),
            diffUrl: buildPairAssetUrl("diff", q.scan_uid, c.scan_uid),
            checkUrl: buildPairAssetUrl("checkerboard", q.scan_uid, c.scan_uid),
          });
        }
      }
    }

    return pairs;
  }

  async function loadImageUrl(url) {
    if (!url) return Promise.resolve({ status: "error", img: null, url: "" });

    if (state.imageLoadCache.has(url)) {
      return state.imageLoadCache.get(url);
    }

    const p = new Promise((resolve) => {
      const img = new Image();
      img.onload = () => resolve({ status: "loaded", img, url });
      img.onerror = () => resolve({ status: "error", img: null, url });
      img.src = url;
    });

    state.imageLoadCache.set(url, p);
    return p;
  }

  async function preloadUrls(
    urls,
    stageLabel,
    doneStart,
    totalAll,
    concurrency = 8,
  ) {
    if (!Array.isArray(urls) || urls.length === 0) {
      return doneStart;
    }

    let completed = 0;
    let cursor = 0;

    async function worker() {
      while (cursor < urls.length) {
        const idx = cursor++;
        const url = urls[idx];
        try {
          await loadImageUrl(url);
        } catch (e) {
          console.warn("[workspace preload] failed:", url, e);
        } finally {
          completed += 1;
          setLoadingStage(stageLabel);
          setLoadingProgress(doneStart + completed, totalAll);
        }
      }
    }

    const workers = [];
    const n = Math.max(1, Math.min(concurrency, urls.length));
    for (let i = 0; i < n; i += 1) {
      workers.push(worker());
    }
    await Promise.all(workers);

    return doneStart + completed;
  }

  function makeWorkingCanvas(w, h) {
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    return canvas;
  }

  async function buildLazyPairAssets(pair, onStepDone = null) {
    if (state.lazyPairAssetCache.has(pair.key)) {
      return state.lazyPairAssetCache.get(pair.key);
    }

    const [qRes, cRes] = await Promise.all([
      loadImageUrl(pair.qUrl),
      loadImageUrl(pair.cUrl),
    ]);

    if (qRes.status !== "loaded" || cRes.status !== "loaded") {
      const failed = { status: "error" };
      state.lazyPairAssetCache.set(pair.key, failed);
      return failed;
    }

    const qImg = qRes.img;
    const cImg = cRes.img;

    const w = qImg.naturalWidth || qImg.width;
    const h = qImg.naturalHeight || qImg.height;

    if (!w || !h) {
      const failed = { status: "error" };
      state.lazyPairAssetCache.set(pair.key, failed);
      return failed;
    }

    if (
      (cImg.naturalWidth || cImg.width) !== w ||
      (cImg.naturalHeight || cImg.height) !== h
    ) {
      const failed = { status: "error" };
      state.lazyPairAssetCache.set(pair.key, failed);
      return failed;
    }

    const qCanvas = makeWorkingCanvas(w, h);
    const cCanvas = makeWorkingCanvas(w, h);
    const diffCanvasWork = makeWorkingCanvas(w, h);
    const checkCanvasWork = makeWorkingCanvas(w, h);

    const qCtx = qCanvas.getContext("2d", { willReadFrequently: true });
    const cCtx = cCanvas.getContext("2d", { willReadFrequently: true });
    const dCtx = diffCanvasWork.getContext("2d", { willReadFrequently: true });
    const chCtx = checkCanvasWork.getContext("2d", {
      willReadFrequently: true,
    });

    if (!qCtx || !cCtx || !dCtx || !chCtx) {
      const failed = { status: "error" };
      state.lazyPairAssetCache.set(pair.key, failed);
      return failed;
    }

    qCtx.drawImage(qImg, 0, 0);
    cCtx.drawImage(cImg, 0, 0);

    const qData = qCtx.getImageData(0, 0, w, h);
    const cData = cCtx.getImageData(0, 0, w, h);

    const qd = qData.data;
    const cd = cData.data;

    const diffOut = dCtx.createImageData(w, h);
    const dd = diffOut.data;

    for (let i = 0; i < dd.length; i += 4) {
      const vq = qd[i];
      const vc = cd[i];
      const d = vq - vc;
      const t = clamp(d / DIFF_MAX, -1, 1);

      let r, g, b;
      if (t >= 0) {
        const k = 1 - t;
        r = 255;
        g = Math.round(255 * k);
        b = Math.round(255 * k);
      } else {
        const k = 1 + t;
        r = Math.round(255 * k);
        g = Math.round(255 * k);
        b = 255;
      }

      dd[i] = r;
      dd[i + 1] = g;
      dd[i + 2] = b;
      dd[i + 3] = 255;
    }

    dCtx.putImageData(diffOut, 0, 0);

    if (onStepDone) {
      await onStepDone();
    }

    const checkOut = chCtx.createImageData(w, h);
    const od = checkOut.data;

    for (let y = 0; y < h; y++) {
      const by = Math.floor(y / CHECK_TILE);
      for (let x = 0; x < w; x++) {
        const bx = Math.floor(x / CHECK_TILE);
        const useQuery = ((bx + by) & 1) === 0;
        const idx = (y * w + x) * 4;
        const src = useQuery ? qd : cd;

        od[idx] = src[idx];
        od[idx + 1] = src[idx + 1];
        od[idx + 2] = src[idx + 2];
        od[idx + 3] = 255;
      }
    }

    chCtx.putImageData(checkOut, 0, 0);

    if (onStepDone) {
      await onStepDone();
    }

    let diffSource = diffCanvasWork;
    let checkSource = checkCanvasWork;

    if (typeof createImageBitmap === "function") {
      try {
        diffSource = await createImageBitmap(diffCanvasWork);
        checkSource = await createImageBitmap(checkCanvasWork);
      } catch (e) {
        // keep canvas fallback
      }
    }

    const built = {
      status: "loaded",
      width: w,
      height: h,
      diffSource,
      checkSource,
    };
    state.lazyPairAssetCache.set(pair.key, built);
    return built;
  }

  async function preloadLazyPairAssets(
    pairs,
    stageLabel,
    doneStart,
    totalAll,
    concurrency = 4,
  ) {
    if (!Array.isArray(pairs) || pairs.length === 0) {
      return doneStart;
    }

    let completed = 0;
    let cursor = 0;
    let lastUiTs = 0;

    async function stepDone() {
      completed += 1;

      const now = performance.now();
      if (now - lastUiTs >= 33 || doneStart + completed >= totalAll) {
        lastUiTs = now;
        setLoadingStage(stageLabel);
        setLoadingProgress(doneStart + completed, totalAll);
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
    }

    async function worker() {
      while (cursor < pairs.length) {
        const idx = cursor++;
        const pair = pairs[idx];
        try {
          await buildLazyPairAssets(pair, stepDone);
        } catch (e) {
          console.warn("[workspace lazy asset build] failed:", pair.key, e);
          await stepDone();
          await stepDone();
        }
      }
    }

    const workers = [];
    const n = Math.max(1, Math.min(concurrency, pairs.length));
    for (let i = 0; i < n; i += 1) {
      workers.push(worker());
    }

    await Promise.all(workers);

    setLoadingStage(stageLabel);
    setLoadingProgress(doneStart + completed, totalAll);
    return doneStart + completed;
  }

  async function preloadWorkspaceAssets() {
    setLoadingStage("Preparing workspace data...");
    setLoadingProgress(0, 1);

    const pngUrls = collectPngUrls();
    const allPairs = collectAllPairs();
    const totalAll = pngUrls.length + allPairs.length * 2;

    if (totalAll === 0) {
      setLoadingProgress(1, 1);
      return;
    }

    let done = 0;

    done = await preloadUrls(
      pngUrls,
      "Preloading PNG images...",
      done,
      totalAll,
      8,
    );

    if (state.reviewMode === "precompute") {
      const pairUrls = [];
      for (const pair of allPairs) {
        pairUrls.push(pair.diffUrl, pair.checkUrl);
      }
      done = await preloadUrls(
        pairUrls,
        "Preloading review assets...",
        done,
        totalAll,
        8,
      );
    } else {
      done = await preloadLazyPairAssets(
        allPairs,
        "Computing review assets...",
        done,
        totalAll,
        4,
      );
    }

    setLoadingProgress(done, totalAll);
  }

  function renderSubjectsView() {
    showView("subjects");
    updateTitle();

    subjectsDataset.textContent = state.dataset;
    subjectsProgressText.textContent = `${state.doneSubjects} / ${state.totalSubjects} subjects done`;

    subjectsList.innerHTML = "";

    for (const row of state.subjects) {
      const subjectId = safeStr(row.subject_id);
      const donePairs = Number(row.done_pairs) || 0;
      const totalPairs = Number(row.total_pairs) || 0;
      const isDone = !!row.is_done;

      const item = document.createElement("div");
      item.className = `list-item ${isDone ? "done" : ""}`;
      item.setAttribute("role", "button");
      item.setAttribute("tabindex", "0");
      item.dataset.subjectId = subjectId;

      item.innerHTML = `
        <div class="list-main">
          <div class="list-title">${subjectId}</div>
          <div class="list-subtitle">${donePairs}/${totalPairs} pairs done</div>
        </div>
        <div class="list-right">
          ${
            isDone
              ? '<span class="badge badge-done">Done</span>'
              : '<span class="badge badge-todo">Todo</span>'
          }
        </div>
      `;

      subjectsList.appendChild(item);
    }
  }

  function initQuerySelect() {
    const subjectData = currentSubjectData();
    querySel.innerHTML = "";

    subjectData.forEach((qb, idx) => {
      const q = qb.query;
      const ses = fmtSession(q.session_id);
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = `Q${idx + 1}: ${ses}`;
      querySel.appendChild(opt);
    });
  }

  function initialQueryIdxFromURL(total) {
    const q = getQueryParam("q");
    if (q === null) return 0;

    const n = parseInt(q, 10);
    if (!Number.isFinite(n)) throw new Error(`Bad q param: ${q}`);

    if (n === -1) return total - 1;
    if (n < 0) return 0;
    if (n >= total) return total - 1;
    return n;
  }

  function enterSubjectsView() {
    renderSubjectsView();
    requestAnimationFrame(() => {
      window.scrollTo(0, state.subjectsScrollY || 0);
    });
  }

  function showMissing(canvasEl, missingEl, msg) {
    setHidden(canvasEl, true);
    setHidden(missingEl, false);
    missingEl.textContent = msg;
  }

  function showCanvas(canvasEl, missingEl) {
    setHidden(missingEl, true);
    setHidden(canvasEl, false);
  }

  function showPanelLoading(canvasEl, missingEl, msg) {
    setHidden(canvasEl, true);
    setHidden(missingEl, false);
    missingEl.textContent = msg;
  }

  function drawLoadedImageToCanvas(img, canvasEl, ctx, missingEl) {
    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    if (!w || !h) {
      showMissing(canvasEl, missingEl, "(Unavailable)");
      return;
    }

    canvasEl.width = w;
    canvasEl.height = h;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0);
    showCanvas(canvasEl, missingEl);
  }

  function drawCachedSourceToCanvas(
    source,
    width,
    height,
    canvasEl,
    ctx,
    missingEl,
  ) {
    if (!source || !width || !height) {
      showMissing(canvasEl, missingEl, "(Unavailable)");
      return;
    }

    canvasEl.width = width;
    canvasEl.height = height;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(source, 0, 0);
    showCanvas(canvasEl, missingEl);
  }

  function setImgOrMissing(imgEl, missingEl, url) {
    if (imgEl.dataset.currentUrl === url) return;

    imgEl.dataset.currentUrl = url;
    setHidden(missingEl, true);
    setHidden(imgEl, false);

    imgEl.onload = () => {
      setHidden(missingEl, true);
      setHidden(imgEl, false);
    };

    imgEl.onerror = () => {
      setHidden(imgEl, true);
      setHidden(missingEl, false);
    };

    imgEl.src = url;
  }

  async function renderDerivedPanels(version, q, c) {
    if (!c) {
      showMissing(diffCanvas, diffMissing, "(Diff unavailable)");
      showMissing(checkCanvas, checkMissing, "(Checkerboard unavailable)");
      return;
    }

    if (state.reviewMode === "precompute") {
      const diffUrl = buildPairAssetUrl("diff", q.scan_uid, c.scan_uid);
      const checkUrl = buildPairAssetUrl(
        "checkerboard",
        q.scan_uid,
        c.scan_uid,
      );

      const [diffRes, checkRes] = await Promise.all([
        loadImageUrl(diffUrl),
        loadImageUrl(checkUrl),
      ]);

      if (version !== state.renderVersion) return;

      if (diffRes.status === "loaded") {
        drawLoadedImageToCanvas(diffRes.img, diffCanvas, diffCtx, diffMissing);
      } else {
        showMissing(diffCanvas, diffMissing, "(Diff unavailable)");
      }

      if (checkRes.status === "loaded") {
        drawLoadedImageToCanvas(
          checkRes.img,
          checkCanvas,
          checkCtx,
          checkMissing,
        );
      } else {
        showMissing(checkCanvas, checkMissing, "(Checkerboard unavailable)");
      }

      return;
    }

    const cached = state.lazyPairAssetCache.get(getPairKey(c));
    if (!cached || cached.status !== "loaded") {
      if (version !== state.renderVersion) return;
      showMissing(diffCanvas, diffMissing, "(Diff unavailable)");
      showMissing(checkCanvas, checkMissing, "(Checkerboard unavailable)");
      return;
    }

    if (version !== state.renderVersion) return;

    drawCachedSourceToCanvas(
      cached.diffSource,
      cached.width,
      cached.height,
      diffCanvas,
      diffCtx,
      diffMissing,
    );
    drawCachedSourceToCanvas(
      cached.checkSource,
      cached.width,
      cached.height,
      checkCanvas,
      checkCtx,
      checkMissing,
    );
  }

  function renderCurrentPair() {
    const version = ++state.renderVersion;

    const qb = currentQueryBlock();
    const q = qb.query;
    const cands = currentCandidates();
    const c = currentCandidate();

    const total = cands.length;
    const idx1 =
      total > 0 ? (((state.candIdx % total) + total) % total) + 1 : 0;

    candCounter.textContent = total > 0 ? `${idx1} / ${total}` : "0 / 0";
    diffCounter.textContent = candCounter.textContent;
    checkCounter.textContent = candCounter.textContent;

    queryMeta.textContent = metaText(q, false);
    setImgOrMissing(queryImg, queryMissing, buildPngUrl(q));

    if (!c) {
      candMeta.textContent = "No candidates for this query.\n\u00A0";
      setHidden(candImg, true);
      setHidden(candMissing, false);
      candMissing.textContent = "(No candidates)";
      applyDecisionToUI(null);
      showMissing(diffCanvas, diffMissing, "(Diff unavailable)");
      showMissing(checkCanvas, checkMissing, "(Checkerboard unavailable)");
      return;
    }

    candMeta.textContent = metaText(c, true);
    setImgOrMissing(candImg, candMissing, buildPngUrl(c));

    const entry = getOrCreateDecisionEntry(q, c);
    applyDecisionToUI(decisionToPlain(entry));

    showPanelLoading(diffCanvas, diffMissing, "(Loading...)");
    showPanelLoading(checkCanvas, checkMissing, "(Loading...)");

    void renderDerivedPanels(version, q, c);
  }

  function enterReviewView(
    subjectId,
    { queryIdx = 0, scrollMode = "preserve" } = {},
  ) {
    subjectId = safeStr(subjectId).trim();
    if (!subjectId) {
      throw new Error("enterReviewView missing subjectId");
    }
    if (!state.reviewBySubject[subjectId]) {
      throw new Error(`Missing subject payload: ${subjectId}`);
    }

    if (scrollMode === "top") {
      window.scrollTo(0, 0);
    }

    state.currentSubjectId = subjectId;
    state.queryIdx = queryIdx;
    state.candIdx = 0;

    reviewDataset.textContent = state.dataset;
    reviewSubjectId.textContent = subjectId;

    showView("review");
    updateTitle();

    initQuerySelect();
    setSelectValueStrict(querySel, String(state.queryIdx));
    renderCurrentPair();

    if (scrollMode === "top") {
      window.scrollTo(0, 0);
    }
  }

  function moveCandidate(delta) {
    const cands = currentCandidates();
    if (!cands || cands.length === 0) return;

    captureCurrentPairFromUI();

    state.candIdx = (state.candIdx + delta) % cands.length;
    if (state.candIdx < 0) state.candIdx += cands.length;

    renderCurrentPair();
  }

  function changeQuery(newIdx) {
    captureCurrentPairFromUI();

    const total = currentSubjectData().length;
    if (newIdx < 0 || newIdx >= total) {
      throw new Error(`changeQuery out of range: ${newIdx}`);
    }

    state.queryIdx = newIdx;
    state.candIdx = 0;

    setSelectValueStrict(querySel, String(state.queryIdx));
    renderCurrentPair();
  }

  function moveQuery(delta) {
    captureCurrentPairFromUI();

    const totalQueries = currentSubjectData().length;

    if (delta < 0) {
      if (state.queryIdx > 0) {
        changeQuery(state.queryIdx - 1);
        return;
      }

      const nav = currentSubjectNav();
      if (nav.prev) {
        const prevTotal = (state.reviewBySubject[nav.prev]?.subject_data || [])
          .length;
        const newQueryIdx = prevTotal > 0 ? prevTotal - 1 : 0;
        enterReviewView(nav.prev, {
          queryIdx: newQueryIdx,
          scrollMode: "preserve",
        });
      }
      return;
    }

    if (delta > 0) {
      if (state.queryIdx < totalQueries - 1) {
        changeQuery(state.queryIdx + 1);
        return;
      }

      const nav = currentSubjectNav();
      if (nav.next) {
        enterReviewView(nav.next, {
          queryIdx: 0,
          scrollMode: "preserve",
        });
      }
    }
  }

  async function backToSubjects() {
    captureCurrentPairFromUI();
    await flushAllDirtyPairs();
    await fetchWorkspaceDataset();
    enterSubjectsView();
  }

  function backToDatasets() {
    captureCurrentPairFromUI();
    void flushAllDirtyPairs().finally(() => {
      window.location.assign(DATASETS_INDEX_URL);
    });
  }

  async function bootstrapWorkspace() {
    showView("loading");
    updateTitle();

    setLoadingStage("Loading dataset structure...");
    setLoadingProgress(0, 1);

    await fetchWorkspaceDataset();
    await preloadWorkspaceAssets();

    const initialView = safeStr(BOOTSTRAP.initial_view).trim().toLowerCase();
    const initialSubjectId = safeStr(BOOTSTRAP.initial_subject_id).trim();

    if (initialView === "review" && initialSubjectId) {
      const total = (
        state.reviewBySubject[initialSubjectId]?.subject_data || []
      ).length;
      const initialQueryIdx = initialQueryIdxFromURL(total);

      state.subjectsScrollY = 0;
      enterReviewView(initialSubjectId, {
        queryIdx: initialQueryIdx,
        scrollMode: "top",
      });
    } else {
      enterSubjectsView();
    }
  }

  subjectsList.addEventListener("click", (e) => {
    const item = e.target.closest("[data-subject-id]");
    if (!item) return;

    const subjectId = safeStr(item.dataset.subjectId).trim();
    if (!subjectId) return;

    saveSubjectsScroll();
    enterReviewView(subjectId, {
      queryIdx: 0,
      scrollMode: "top",
    });
  });

  subjectsList.addEventListener("keydown", (e) => {
    const item = e.target.closest("[data-subject-id]");
    if (!item) return;

    if (e.key !== "Enter" && e.key !== " ") return;
    e.preventDefault();

    const subjectId = safeStr(item.dataset.subjectId).trim();
    if (!subjectId) return;

    saveSubjectsScroll();
    enterReviewView(subjectId, {
      queryIdx: 0,
      scrollMode: "top",
    });
  });

  btnBackToSubjects.addEventListener("click", () => {
    void backToSubjects().catch((err) => {
      console.error(err);
    });
  });

  btnBackToDatasets.addEventListener("click", () => {
    backToDatasets();
  });

  querySel.addEventListener("change", () => {
    const idx = parseInt(querySel.value, 10);
    if (!Number.isFinite(idx)) {
      throw new Error(`Bad querySel value: ${querySel.value}`);
    }
    changeQuery(idx);
  });

  btnYes.addEventListener("click", () => {
    setDecisionLocal("yes");
  });

  btnNo.addEventListener("click", () => {
    setDecisionLocal("no");
  });

  btnMaybe.addEventListener("click", () => {
    setDecisionLocal("maybe");
  });

  reasonBox.addEventListener("input", () => {
    captureCurrentPairFromUI();
  });

  reasonBox.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.shiftKey) return;
    e.preventDefault();
    captureCurrentPairFromUI();
    scheduleFlushDirtyPairs(0);
  });

  document.addEventListener("keydown", (e) => {
    const tag =
      e.target && e.target.tagName ? e.target.tagName.toLowerCase() : "";
    const isTyping = tag === "textarea" || tag === "input" || tag === "select";
    if (isTyping) return;
    if (state.currentView !== "review") return;

    if (e.key === "ArrowLeft") {
      e.preventDefault();
      moveCandidate(-1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      moveCandidate(+1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      moveQuery(-1);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      moveQuery(+1);
    }
  });

  window.addEventListener("beforeunload", () => {
    captureCurrentPairFromUI();
  });

  bootstrapWorkspace().catch((err) => {
    console.error(err);
    showView("loading");
    setLoadingStage("Failed to load workspace");
    loadingProgressText.textContent = safeStr(
      err && err.message ? err.message : err,
    );
    loadingBarFill.style.width = "100%";
  });
})();
