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

  function getQueryParam(name) {
    return new URLSearchParams(window.location.search).get(name);
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

  function withParams(href, paramsObj) {
    const u = new URL(href, window.location.origin);
    for (const [k, v] of Object.entries(paramsObj)) {
      if (v === null || v === undefined || v === "") continue;
      u.searchParams.set(k, String(v));
    }
    return (
      u.pathname +
      (u.searchParams.toString() ? `?${u.searchParams.toString()}` : "")
    );
  }

  function clamp(x, lo, hi) {
    return x < lo ? lo : x > hi ? hi : x;
  }

  const NAV_LOADING_KEY = "navLoading";
  const REVIEW_SCROLL_KEY = "review:scrollY";

  const overlay = $("globalLoadingOverlay");

  function isNavLoading() {
    return sessionStorage.getItem(NAV_LOADING_KEY) === "1";
  }

  function beginNavLoading() {
    sessionStorage.setItem(NAV_LOADING_KEY, "1");
    document.documentElement.classList.add("nav-loading");
    overlay.style.display = "flex";
  }

  function endNavLoading() {
    sessionStorage.removeItem(NAV_LOADING_KEY);
    document.documentElement.classList.remove("nav-loading");
    overlay.style.display = "none";
  }

  function saveReviewScrollY() {
    sessionStorage.setItem(REVIEW_SCROLL_KEY, String(window.scrollY || 0));
  }

  function restoreReviewScrollYNow() {
    const raw = sessionStorage.getItem(REVIEW_SCROLL_KEY);
    if (raw === null) return;

    const yy = parseInt(raw, 10);
    if (!Number.isFinite(yy) || yy < 0) throw new Error(`Bad scrollY: ${raw}`);

    window.scrollTo(0, yy);
    sessionStorage.removeItem(REVIEW_SCROLL_KEY);
  }

  function waitForImage(imgEl) {
    if (imgEl.style.display === "none") return Promise.resolve("hidden");

    if (imgEl.complete) {
      return Promise.resolve(imgEl.naturalWidth > 0 ? "loaded" : "error");
    }

    return new Promise((resolve) => {
      imgEl.addEventListener("load", () => resolve("loaded"), { once: true });
      imgEl.addEventListener("error", () => resolve("error"), { once: true });
    });
  }

  const SUBJECT_DATA = readInjectedJSON("subject-data");
  const SUBJECT_NAV = readInjectedJSON("subject-nav");

  if (!Array.isArray(SUBJECT_DATA) || SUBJECT_DATA.length === 0) {
    throw new Error("subject-data is empty or not an array");
  }

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

  const btnPrev = $("btnPrev");
  const btnNext = $("btnNext");

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

  let queryIdx = 0;
  let candIdx = 0;
  let busy = false;

  function currentQueryBlock() {
    const qb = SUBJECT_DATA[queryIdx];
    if (!qb) throw new Error(`queryIdx out of range: ${queryIdx}`);
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
    const i = ((candIdx % cands.length) + cands.length) % cands.length;
    return cands[i];
  }

  function pairPayload(a, b) {
    return { a, b };
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
    const q = currentQuery();
    const c = currentCandidate();
    if (!c) return null;
    return await savePairState(q, c);
  }

  async function setDecision(status) {
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

  const DIFF_MAX = 64;
  const CHECK_TILE = 64;

  function showMissing(canvasEl, missingEl, msg) {
    canvasEl.style.display = "none";
    missingEl.style.display = "block";
    missingEl.textContent = msg;
  }

  function showCanvas(canvasEl, missingEl) {
    missingEl.style.display = "none";
    canvasEl.style.display = "block";
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

  function computeAndRenderDiffAndChecker() {
    if (!queryImg.complete || !candImg.complete) return;

    if (queryImg.style.display === "none" || candImg.style.display === "none") {
      showMissing(diffCanvas, diffMissing, "(Diff unavailable)");
      showMissing(checkCanvas, checkMissing, "(Checkerboard unavailable)");
      return;
    }

    if (queryImg.naturalWidth <= 0 || queryImg.naturalHeight <= 0) {
      throw new Error("queryImg has no natural size (failed to load?)");
    }
    if (candImg.naturalWidth <= 0 || candImg.naturalHeight <= 0) {
      throw new Error("candImg has no natural size (failed to load?)");
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
  }

  function setImgOrMissing(imgEl, missingEl, url) {
    missingEl.style.display = "none";
    imgEl.style.display = "block";

    imgEl.onload = () => {
      computeAndRenderDiffAndChecker();
    };

    imgEl.onerror = () => {
      imgEl.style.display = "none";
      missingEl.style.display = "block";
      computeAndRenderDiffAndChecker();
    };

    imgEl.src = url;
  }

  async function renderAll() {
    const qb = currentQueryBlock();
    const q = qb.query;
    const cands = currentCandidates();
    const c = currentCandidate();

    const total = cands.length;
    const idx1 = total > 0 ? (((candIdx % total) + total) % total) + 1 : 0;
    candCounter.textContent = total > 0 ? `${idx1} / ${total}` : "0 / 0";
    diffCounter.textContent = candCounter.textContent;
    checkCounter.textContent = candCounter.textContent;

    queryMeta.textContent = metaText(q, false);
    setImgOrMissing(queryImg, queryMissing, buildPngUrl(q));

    if (c) {
      candMeta.textContent = metaText(c, true);
      setImgOrMissing(candImg, candMissing, buildPngUrl(c));

      const res = await ensureViewed(q, c);
      applyDecisionToUI(res.decision || null);
    } else {
      candMeta.textContent = "No candidates for this query.\n\u00A0";
      candImg.style.display = "none";
      candMissing.style.display = "block";
      candMissing.textContent = "(No candidates)";
      applyDecisionToUI(null);

      showMissing(diffCanvas, diffMissing, "(Diff unavailable)");
      showMissing(checkCanvas, checkMissing, "(Checkerboard unavailable)");
    }

    computeAndRenderDiffAndChecker();
  }

  async function finalizeIfNavLoading() {
    if (!isNavLoading()) return;

    await waitForImage(queryImg);
    await waitForImage(candImg);

    computeAndRenderDiffAndChecker();

    restoreReviewScrollYNow();

    endNavLoading();
  }

  async function moveCandidate(delta) {
    const c = currentCandidate();
    const cands = currentCandidates();
    if (!c || cands.length === 0) return;

    await saveCurrentPairState();

    candIdx = (candIdx + delta) % cands.length;
    if (candIdx < 0) candIdx += cands.length;

    await renderAll();
  }

  async function changeQuery(newIdx) {
    const oldC = currentCandidate();
    if (oldC) await saveCurrentPairState();

    if (newIdx < 0 || newIdx >= SUBJECT_DATA.length) {
      throw new Error(`changeQuery out of range: ${newIdx}`);
    }

    queryIdx = newIdx;
    candIdx = 0;

    setSelectValueStrict(querySel, String(queryIdx));
    await renderAll();
  }

  async function navigateToSubject(
    href,
    { goLastQuery } = { goLastQuery: false },
  ) {
    const c = currentCandidate();
    if (c) await saveCurrentPairState();

    beginNavLoading();
    saveReviewScrollY();

    const target = withParams(href, {
      q: goLastQuery ? -1 : null,
    });

    requestAnimationFrame(() => {
      window.location.replace(target);
    });
  }

  async function moveQuery(delta) {
    if (delta < 0) {
      if (queryIdx > 0) {
        await changeQuery(queryIdx - 1);
        return;
      }
      if (SUBJECT_NAV.prev) {
        await navigateToSubject(SUBJECT_NAV.prev, { goLastQuery: true });
      }
      return;
    }

    if (delta > 0) {
      if (queryIdx < SUBJECT_DATA.length - 1) {
        await changeQuery(queryIdx + 1);
        return;
      }
      if (SUBJECT_NAV.next) {
        await navigateToSubject(SUBJECT_NAV.next, { goLastQuery: false });
      }
    }
  }

  function initQuerySelect() {
    querySel.innerHTML = "";
    SUBJECT_DATA.forEach((qb, idx) => {
      const q = qb.query;
      const ses = fmtSession(q.session_id);
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = `Q${idx + 1}: ${ses}`;
      querySel.appendChild(opt);
    });
  }

  function initialQueryIdxFromURL() {
    const total = SUBJECT_DATA.length;
    const q = getQueryParam("q");
    if (q === null) return 0;

    const n = parseInt(q, 10);
    if (!Number.isFinite(n)) throw new Error(`Bad q param: ${q}`);

    if (n === -1) return total - 1;
    if (n < 0) return 0;
    if (n >= total) return total - 1;
    return n;
  }

  async function guarded(fn) {
    if (busy) return;
    busy = true;
    try {
      await fn();
    } finally {
      busy = false;
    }
  }

  querySel.addEventListener("change", () =>
    guarded(async () => {
      const idx = parseInt(querySel.value, 10);
      if (!Number.isFinite(idx)) {
        throw new Error(`Bad querySel value: ${querySel.value}`);
      }
      await changeQuery(idx);
    }),
  );

  btnPrev.addEventListener("click", () =>
    guarded(async () => {
      await moveCandidate(-1);
    }),
  );

  btnNext.addEventListener("click", () =>
    guarded(async () => {
      await moveCandidate(+1);
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

  (async () => {
    initQuerySelect();

    queryIdx = initialQueryIdxFromURL();
    setSelectValueStrict(querySel, String(queryIdx));
    setSelected("no");
    reasonBox.value = "";

    await renderAll();
    await finalizeIfNavLoading();

    computeAndRenderDiffAndChecker();
  })();
})();
