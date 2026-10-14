(() => {
  const $ = (id) => document.getElementById(id);

  const fileInput = $("fileInput");
  const dropzone = $("dropzone");
  const dropEmpty = $("dropEmpty");
  const previewImg = $("previewImg");
  const conf = $("conf");
  const confOut = $("confOut");
  const ocrToggle = $("ocrToggle");
  const runBtn = $("runBtn");
  const errorBox = $("error");

  const emptyState = $("empty");
  const loading = $("loading");
  const loadingText = $("loadingText");
  const results = $("results");

  let selectedFile = null;

  // ─── Helpers ───
  const showError = (msg) => {
    errorBox.textContent = msg;
    errorBox.hidden = !msg;
  };

  const setState = (state) => {
    emptyState.hidden = state !== "empty";
    loading.hidden = state !== "loading";
    results.hidden = state !== "results";
  };

  const setFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showError("Please choose an image file.");
      return;
    }
    showError("");
    selectedFile = file;
    previewImg.src = URL.createObjectURL(file);
    previewImg.hidden = false;
    dropEmpty.hidden = true;
    runBtn.disabled = false;
  };

  // ─── Upload: click, drag & drop ───
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-over");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-over");
    })
  );
  dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  // ─── Slider ───
  conf.addEventListener("input", () => {
    confOut.textContent = Number(conf.value).toFixed(2);
  });

  // ─── Run detection ───
  runBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    showError("");
    runBtn.disabled = true;
    loadingText.textContent = ocrToggle.checked
      ? "Detecting objects and reading text… the first run loads the OCR model and can take a minute."
      : "Detecting objects…";
    setState("loading");

    const form = new FormData();
    form.append("image", selectedFile);
    form.append("conf", conf.value);
    form.append("ocr", ocrToggle.checked ? "true" : "false");

    try {
      const res = await fetch("/api/predict", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong on the server.");
      render(data);
    } catch (err) {
      showError(err.message);
      setState("empty");
    } finally {
      runBtn.disabled = false;
    }
  });

  // ─── Render results ───
  function render(data) {
    $("resultImg").src = data.image;
    $("statCount").textContent = data.count;
    $("statClasses").textContent = Object.keys(data.summary).length;

    const withText = data.detections.filter((d) => d.text && d.text.length).length;
    $("statText").textContent = data.ocr_enabled ? withText : "–";

    // Chips
    const chips = $("chips");
    chips.innerHTML = "";
    Object.entries(data.summary)
      .sort((a, b) => b[1] - a[1])
      .forEach(([name, n]) => {
        const li = document.createElement("li");
        li.append(name + " ");
        const b = document.createElement("b");
        b.textContent = n;
        li.append(b);
        chips.append(li);
      });
    if (!data.count) {
      chips.innerHTML = '<li class="muted">No objects found. Try lowering the confidence threshold.</li>';
    }

    // Table
    const rows = $("rows");
    rows.innerHTML = "";
    data.detections.forEach((d) => {
      const tr = document.createElement("tr");

      const tdId = document.createElement("td");
      tdId.className = "num";
      tdId.textContent = d.id;

      const tdClass = document.createElement("td");
      tdClass.textContent = d.class;

      const tdConf = document.createElement("td");
      const pct = (d.confidence * 100).toFixed(1);
      tdConf.innerHTML = `<div class="conf"><div class="conf__bar"><i style="width:${pct}%"></i></div><span class="num">${pct}%</span></div>`;

      const tdText = document.createElement("td");
      if (!data.ocr_enabled) {
        tdText.innerHTML = '<span class="muted">OCR off</span>';
      } else if (d.text && d.text.length) {
        const wrap = document.createElement("div");
        wrap.className = "text-found";
        d.text.forEach((t) => {
          const s = document.createElement("span");
          s.textContent = t;
          wrap.append(s);
        });
        tdText.append(wrap);
      } else {
        tdText.innerHTML = '<span class="muted">No text</span>';
      }

      tr.append(tdId, tdClass, tdConf, tdText);
      rows.append(tr);
    });

    setState("results");
  }
})();
