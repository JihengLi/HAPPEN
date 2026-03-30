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

  function isHidden(el) {
    return el.classList.contains("is-hidden");
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

  function waitForImage(imgEl) {
    if (isHidden(imgEl)) return Promise.resolve("hidden");

    if (imgEl.complete) {
      return Promise.resolve(imgEl.naturalWidth > 0 ? "loaded" : "error");
    }

    return new Promise((resolve) => {
      imgEl.addEventListener("load", () => resolve("loaded"), { once: true });
      imgEl.addEventListener("error", () => resolve("error"), { once: true });
    });
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

  const scratchCanvas = document.createElement("canvas");
  const scratchCtx = scratchCanvas.getContext("2d", {
    willReadFrequently: true,
  });
  if (!scratchCtx) throw new Error("scratchCanvas.getContext returned null");

  const overlay = document.getElementById("globalLoadingOverlay");
  try {
    sessionStorage.removeItem("navLoading");
  } catch (e) {}
  document.documentElement.classList.remove("nav-loading");
  if (overlay) overlay.classList.add("is-hidden");

  const state = {
    dataset: safeStr(BOOTSTRAP.dataset).trim(),
    reviewMode: safeStr(BOOTSTRAP.review_mode).trim().toLowerCase(),
    subjects: [],
    reviewBySubject: {},
    doneSubjects: 0,
    totalSubjects: 0,

    currentView: "loading", // loading | subjects | review
    currentSubjectId: "",
    queryIdx: 0,
    candIdx: 0,

    subjectsScrollY: 0,

    busy: false,
    renderVersion: 0,

    imageLoadCache: new Map(),
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

  function pairPayload(a, b) {
    return { a, b };
  }

  async function ensureViewed(a, b) {
    const res = await postJSON("/api/view", pairPayload(a, b));
    if (!res.ok) throw new Error("api/view returned not ok");
    return res;
  }

  async function savePairState(a, b) {
    const payload = {
      ...pairPayload(a, b),
      qa_status: currentSelectedStatus(),
      reason: reasonBox.value || "",
    };
    const res = await postJSON("/api/decision", payload);
    if (!res.ok) throw new Error("api/decision returned not ok");
    return res;
  }

  async function saveCurrentPairState() {
    if (state.currentView !== "review") return null;
    const q = currentQuery();
    const c = currentCandidate();
    if (!c) return null;
    return await savePairState(q, c);
  }

  async function setDecision(status) {
    if (state.currentView !== "review") return;

    const q = currentQuery();
    const c = currentCandidate();
    if (!c) return;

    setSelected(status);

    const payload = {
      ...pairPayload(q, c),
      qa_status: status,
      reason: reasonBox.value || "",
    };
    const res = await postJSON("/api/decision", payload);
    if (!res.ok) throw new Error("api/decision returned not ok");

    applyDecisionToUI(res.decision || null);
  }

  async function fetchWorkspaceDataset() {
    const u = new URL(WORKSPACE_API_URL, window.location.origin);
    u.searchParams.set("dataset", state.dataset);

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
    state.doneSubjects = Number(data.done_subjects) || 0;
    state.totalSubjects = Number(data.total_subjects) || 0;
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
        for (const cand of cands) {
          urls.add(buildPngUrl(cand));
        }
      }
    }

    return Array.from(urls);
  }

  function collectPairAssetUrls() {
    if (state.reviewMode !== "precompute") {
      return [];
    }

    const urls = new Set();

    for (const subjectId of Object.keys(state.reviewBySubject)) {
      const payload = state.reviewBySubject[subjectId];
      const subjectData = Array.isArray(payload.subject_data)
        ? payload.subject_data
        : [];

      for (const qb of subjectData) {
        if (!qb.query) continue;
        const q = qb.query;
        const cands = Array.isArray(qb.candidates) ? qb.candidates : [];
        for (const cand of cands) {
          urls.add(buildPairAssetUrl("diff", q.scan_uid, cand.scan_uid));
          urls.add(
            buildPairAssetUrl("checkerboard", q.scan_uid, cand.scan_uid),
          );
        }
      }
    }

    return Array.from(urls);
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

  async function preloadWorkspaceAssets() {
    setLoadingStage("Preparing workspace data...");
    setLoadingProgress(0, 1);

    const pngUrls = collectPngUrls();
    const pairUrls = collectPairAssetUrls();
    const totalAll = pngUrls.length + pairUrls.length;

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

    if (pairUrls.length > 0) {
      done = await preloadUrls(
        pairUrls,
        "Preloading review assets...",
        done,
        totalAll,
        8,
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

  async function enterReviewView(
    subjectId,
    {
      queryIdx = 0,
      scrollMode = "preserve", // "top" | "preserve"
    } = {},
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
    setSelected("no");
    reasonBox.value = "";

    await renderAll();

    if (scrollMode === "top") {
      window.scrollTo(0, 0);
    }
  }

  const DIFF_MAX = 64;
  const CHECK_TILE = 32;

  function showMissing(canvasEl, missingEl, msg) {
    setHidden(canvasEl, true);
    setHidden(missingEl, false);
    missingEl.textContent = msg;
  }

  function showCanvas(canvasEl, missingEl) {
    setHidden(missingEl, true);
    setHidden(canvasEl, false);
  }

  function signedToRGB(t) {
    t = clamp(t, -1, 1);
    if (t >= 0) {
      const k = 1 - t;
      return [255, Math.round(255 * k), Math.round(255 * k)];
    } else {
      const k = 1 + t;
      return [Math.round(255 * k), Math.round(255 * k), 255];
    }
  }

  function readImageData(imgEl, w, h) {
    scratchCanvas.width = w;
    scratchCanvas.height = h;
    scratchCtx.clearRect(0, 0, w, h);
    scratchCtx.drawImage(imgEl, 0, 0);
    return scratchCtx.getImageData(0, 0, w, h);
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

  function computeAndRenderDiffAndCheckerLocal() {
    if (!queryImg.complete || !candImg.complete) return false;

    if (isHidden(queryImg) || isHidden(candImg)) {
      showMissing(diffCanvas, diffMissing, "(Diff unavailable)");
      showMissing(checkCanvas, checkMissing, "(Checkerboard unavailable)");
      return false;
    }

    if (queryImg.naturalWidth <= 0 || queryImg.naturalHeight <= 0) {
      return false;
    }
    if (candImg.naturalWidth <= 0 || candImg.naturalHeight <= 0) {
      return false;
    }

    if (
      queryImg.naturalWidth !== candImg.naturalWidth ||
      queryImg.naturalHeight !== candImg.naturalHeight
    ) {
      throw new Error(
        `Image size mismatch: query=${queryImg.naturalWidth}x${queryImg.naturalHeight}, cand=${candImg.naturalWidth}x${candImg.naturalHeight}`,
      );
    }

    const w = queryImg.naturalWidth;
    const h = queryImg.naturalHeight;

    const q = readImageData(queryImg, w, h);
    const c = readImageData(candImg, w, h);

    const qd = q.data;
    const cd = c.data;

    diffCanvas.width = w;
    diffCanvas.height = h;

    const diffOut = new ImageData(w, h);
    const dd = diffOut.data;

    for (let i = 0; i < dd.length; i += 4) {
      const vq = qd[i];
      const vc = cd[i];
      const d = vq - vc;
      const t = clamp(d / DIFF_MAX, -1, 1);
      const [r, g, b] = signedToRGB(t);

      dd[i] = r;
      dd[i + 1] = g;
      dd[i + 2] = b;
      dd[i + 3] = 255;
    }

    diffCtx.putImageData(diffOut, 0, 0);
    showCanvas(diffCanvas, diffMissing);

    checkCanvas.width = w;
    checkCanvas.height = h;

    const chkOut = new ImageData(w, h);
    const od = chkOut.data;

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

    checkCtx.putImageData(chkOut, 0, 0);
    showCanvas(checkCanvas, checkMissing);
    return true;
  }

  function setImgOrMissing(imgEl, missingEl, url) {
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

    if (state.reviewMode === "lazy") {
      const [qStatus, cStatus] = await Promise.all([
        waitForImage(queryImg),
        waitForImage(candImg),
      ]);

      if (version !== state.renderVersion) return;

      if (qStatus === "loaded" && cStatus === "loaded") {
        const ok = computeAndRenderDiffAndCheckerLocal();
        if (ok) return;
      }

      showMissing(diffCanvas, diffMissing, "(Diff unavailable)");
      showMissing(checkCanvas, checkMissing, "(Checkerboard unavailable)");
      return;
    }

    throw new Error(`Unhandled review_mode: ${state.reviewMode}`);
  }

  async function renderAll() {
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

    if (c) {
      candMeta.textContent = metaText(c, true);
      setImgOrMissing(candImg, candMissing, buildPngUrl(c));

      const res = await ensureViewed(q, c);
      if (version !== state.renderVersion) return;
      applyDecisionToUI(res.decision || null);
    } else {
      candMeta.textContent = "No candidates for this query.\n\u00A0";
      setHidden(candImg, true);
      setHidden(candMissing, false);
      candMissing.textContent = "(No candidates)";
      applyDecisionToUI(null);

      showMissing(diffCanvas, diffMissing, "(Diff unavailable)");
      showMissing(checkCanvas, checkMissing, "(Checkerboard unavailable)");
      return;
    }

    await renderDerivedPanels(version, q, c);
  }

  async function moveCandidate(delta) {
    const c = currentCandidate();
    const cands = currentCandidates();
    if (!c || cands.length === 0) return;

    await saveCurrentPairState();

    state.candIdx = (state.candIdx + delta) % cands.length;
    if (state.candIdx < 0) state.candIdx += cands.length;

    await renderAll();
  }

  async function changeQuery(newIdx) {
    const oldC = currentCandidate();
    if (oldC) await saveCurrentPairState();

    const total = currentSubjectData().length;
    if (newIdx < 0 || newIdx >= total) {
      throw new Error(`changeQuery out of range: ${newIdx}`);
    }

    state.queryIdx = newIdx;
    state.candIdx = 0;

    setSelectValueStrict(querySel, String(state.queryIdx));
    await renderAll();
  }

  async function moveQuery(delta) {
    const totalQueries = currentSubjectData().length;

    if (delta < 0) {
      if (state.queryIdx > 0) {
        await changeQuery(state.queryIdx - 1);
        return;
      }

      const nav = currentSubjectNav();
      if (nav.prev) {
        await saveCurrentPairState();
        const prevTotal = (state.reviewBySubject[nav.prev]?.subject_data || [])
          .length;
        const newQueryIdx = prevTotal > 0 ? prevTotal - 1 : 0;
        await enterReviewView(nav.prev, {
          queryIdx: newQueryIdx,
          scrollMode: "preserve",
        });
      }
      return;
    }

    if (delta > 0) {
      if (state.queryIdx < totalQueries - 1) {
        await changeQuery(state.queryIdx + 1);
        return;
      }

      const nav = currentSubjectNav();
      if (nav.next) {
        await saveCurrentPairState();
        await enterReviewView(nav.next, {
          queryIdx: 0,
          scrollMode: "preserve",
        });
      }
    }
  }

  async function loadImageUrl(url) {
    if (!url) {
      return Promise.resolve({ status: "error", img: null, url: "" });
    }

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

  async function guarded(fn) {
    if (state.busy) return;
    state.busy = true;
    try {
      await fn();
    } finally {
      state.busy = false;
    }
  }

  async function backToSubjects() {
    await saveCurrentPairState();
    await fetchWorkspaceDataset();
    enterSubjectsView();
  }

  function backToDatasets() {
    window.location.assign(DATASETS_INDEX_URL);
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

      await enterReviewView(initialSubjectId, {
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

    guarded(async () => {
      saveSubjectsScroll();
      await enterReviewView(subjectId, {
        queryIdx: 0,
        scrollMode: "top",
      });
    });
  });

  subjectsList.addEventListener("keydown", (e) => {
    const item = e.target.closest("[data-subject-id]");
    if (!item) return;

    if (e.key !== "Enter" && e.key !== " ") return;
    e.preventDefault();

    const subjectId = safeStr(item.dataset.subjectId).trim();
    if (!subjectId) return;

    guarded(async () => {
      saveSubjectsScroll();
      await enterReviewView(subjectId, {
        queryIdx: 0,
        scrollMode: "top",
      });
    });
  });

  btnBackToSubjects.addEventListener("click", () =>
    guarded(async () => {
      await backToSubjects();
    }),
  );

  btnBackToDatasets.addEventListener("click", () => {
    backToDatasets();
  });

  querySel.addEventListener("change", () =>
    guarded(async () => {
      const idx = parseInt(querySel.value, 10);
      if (!Number.isFinite(idx)) {
        throw new Error(`Bad querySel value: ${querySel.value}`);
      }
      await changeQuery(idx);
    }),
  );

  btnYes.addEventListener("click", () =>
    guarded(async () => {
      await setDecision("yes");
    }),
  );

  btnNo.addEventListener("click", () =>
    guarded(async () => {
      await setDecision("no");
    }),
  );

  btnMaybe.addEventListener("click", () =>
    guarded(async () => {
      await setDecision("maybe");
    }),
  );

  reasonBox.addEventListener("keydown", (e) =>
    guarded(async () => {
      if (e.key !== "Enter") return;
      if (e.shiftKey) {
        return;
      }

      e.preventDefault();

      const res = await saveCurrentPairState();
      if (res && res.decision) {
        applyDecisionToUI(res.decision);
      }
    }),
  );

  document.addEventListener("keydown", (e) => {
    const tag =
      e.target && e.target.tagName ? e.target.tagName.toLowerCase() : "";
    const isTyping = tag === "textarea" || tag === "input" || tag === "select";
    if (isTyping) return;

    if (state.currentView !== "review") return;

    if (e.key === "ArrowLeft") {
      e.preventDefault();
      guarded(async () => {
        await moveCandidate(-1);
      });
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      guarded(async () => {
        await moveCandidate(+1);
      });
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      guarded(async () => {
        await moveQuery(-1);
      });
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      guarded(async () => {
        await moveQuery(+1);
      });
    }
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
