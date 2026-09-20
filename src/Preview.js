/**
 * src/Preview.js
 * Interactive UI Preview for Vessert Tensor Engine & Ve HTTP Framework.
 * Auto-mounts in browser DOM to provide a live demonstration of
 * tensor math, neural model inference, HTTP routing, and DAG architecture.
 */

import { Vessert, Dtype, Device, config, ve } from './index.js';
import { predict, info } from './app/model.js';
import { mountRoutes } from './app/routes.js';

/**
 * Initializes and mounts the interactive Preview into the target container.
 * @param {HTMLElement} [container]
 */
export function renderPreview(container = document.getElementById('app') || document.body) {
  if (!container) return;

  container.innerHTML = `
    <div class="pv-root">
      <!-- HEADER -->
      <header class="pv-header">
        <div class="pv-header-left">
          <div class="pv-logo">
            <span class="pv-logo-symbol">◆</span>
            <div>
              <h1 class="pv-title">Vessert & Ve Engine</h1>
              <p class="pv-subtitle">Tensor Math & Micro-HTTP Framework Preview</p>
            </div>
          </div>
        </div>
        <div class="pv-badges">
          <span class="pv-badge pv-badge-green"><span class="pv-dot"></span> System Operational</span>
          <span class="pv-badge">Node.js + Browser</span>
          <span class="pv-badge">DAG Verified</span>
        </div>
      </header>

      <!-- NAVIGATION TABS -->
      <nav class="pv-nav" id="pv-nav">
        <button class="pv-tab active" data-tab="tab-tensors" id="btn-tab-tensors">Tensor Engine (Vessert)</button>
        <button class="pv-tab" data-tab="tab-model" id="btn-tab-model">Model Inference</button>
        <button class="pv-tab" data-tab="tab-ve" id="btn-tab-ve">HTTP Framework (Ve)</button>
        <button class="pv-tab" data-tab="tab-dag" id="btn-tab-dag">Dependency Architecture (DAG)</button>
      </nav>

      <!-- TAB CONTENT PANELS -->
      <main class="pv-main">
        <!-- TAB 1: TENSOR ENGINE -->
        <section id="tab-tensors" class="pv-panel active">
          <div class="pv-grid-2">
            <!-- Card: Interactive Matmul -->
            <div class="pv-card">
              <div class="pv-card-header">
                <h3>Matrix Multiplication (Matmul 2x2 @ 2x2)</h3>
                <span class="pv-chip">Linear Algebra</span>
              </div>
              <p class="pv-desc">Kalkulasi perkalian matriks menggunakan alur numerik Vessert berkecepatan tinggi.</p>
              
              <div class="pv-matrix-grid">
                <div class="pv-matrix-box">
                  <label>Matrix A (2x2)</label>
                  <div class="pv-inputs-2x2">
                    <input type="number" id="ma00" value="1" step="0.5" class="pv-input">
                    <input type="number" id="ma01" value="2" step="0.5" class="pv-input">
                    <input type="number" id="ma10" value="3" step="0.5" class="pv-input">
                    <input type="number" id="ma11" value="4" step="0.5" class="pv-input">
                  </div>
                </div>
                <div class="pv-matrix-op">✕</div>
                <div class="pv-matrix-box">
                  <label>Matrix B (2x2)</label>
                  <div class="pv-inputs-2x2">
                    <input type="number" id="mb00" value="5" step="0.5" class="pv-input">
                    <input type="number" id="mb01" value="6" step="0.5" class="pv-input">
                    <input type="number" id="mb10" value="7" step="0.5" class="pv-input">
                    <input type="number" id="mb11" value="8" step="0.5" class="pv-input">
                  </div>
                </div>
              </div>

              <div class="pv-actions">
                <button id="btn-calc-matmul" class="pv-btn pv-btn-primary">Hitung A @ B</button>
                <button id="btn-rand-matmul" class="pv-btn pv-btn-secondary">Randomize</button>
              </div>

              <div class="pv-result-box">
                <div class="pv-res-title">Hasil Tensor C = A.matmul(B):</div>
                <div id="res-matmul" class="pv-code-block">Menunggu perhitungan...</div>
              </div>
            </div>

            <!-- Card: Elementwise & Reductions -->
            <div class="pv-card">
              <div class="pv-card-header">
                <h3>Tensor Factory & Reductions</h3>
                <span class="pv-chip">Operator Facade</span>
              </div>
              <p class="pv-desc">Uji coba pembuatan tensor langsung ('zeros', 'ones', 'arange', 'eye', 'rand').</p>
              
              <div class="pv-actions">
                <button id="btn-tensor-zeros" class="pv-btn pv-btn-outline">zeros([3, 3])</button>
                <button id="btn-tensor-ones" class="pv-btn pv-btn-outline">ones([2, 4])</button>
                <button id="btn-tensor-eye" class="pv-btn pv-btn-outline">eye(4)</button>
                <button id="btn-tensor-arange" class="pv-btn pv-btn-outline">arange(0, 10, 2)</button>
                <button id="btn-tensor-randn" class="pv-btn pv-btn-outline">randn([3, 3])</button>
              </div>

              <div class="pv-result-box" style="margin-top: 16px;">
                <div class="pv-res-title">Inspeksi Data & Metadata:</div>
                <div id="res-factory" class="pv-code-block">Klik salah satu tombol di atas untuk generate tensor.</div>
              </div>
            </div>
          </div>
        </section>

        <!-- TAB 2: MODEL INFERENCE -->
        <section id="tab-model" class="pv-panel">
          <div class="pv-card">
            <div class="pv-card-header">
              <h3>Live Neural Softmax Classifier (src/app/model.js)</h3>
              <span class="pv-chip pv-chip-blue">Inference Engine</span>
            </div>
            <p class="pv-desc">Model klasifikasi 2-kelas dengan bobot dan bias internal menggunakan layer Linear + Softmax via Vessert.</p>

            <div class="pv-model-controls">
              <div class="pv-slider-group">
                <div class="pv-slider-header">
                  <span>Feature 1 (X₁):</span>
                  <span id="val-f1" class="pv-val-num">0.50</span>
                </div>
                <input type="range" id="slider-f1" min="-2" max="2" step="0.05" value="0.5" class="pv-range">
              </div>

              <div class="pv-slider-group">
                <div class="pv-slider-header">
                  <span>Feature 2 (X₂):</span>
                  <span id="val-f2" class="pv-val-num">-0.20</span>
                </div>
                <input type="range" id="slider-f2" min="-2" max="2" step="0.05" value="-0.2" class="pv-range">
              </div>

              <div class="pv-slider-group">
                <div class="pv-slider-header">
                  <span>Feature 3 (X₃):</span>
                  <span id="val-f3" class="pv-val-num">0.90</span>
                </div>
                <input type="range" id="slider-f3" min="-2" max="2" step="0.05" value="0.9" class="pv-range">
              </div>
            </div>

            <div class="pv-prob-container">
              <h4>Distribusi Probabilitas Softmax:</h4>
              <div class="pv-prob-row">
                <div class="pv-prob-labels">
                  <span>Class 0 (Negative)</span>
                  <span id="label-p0">50.0%</span>
                </div>
                <div class="pv-bar-track">
                  <div id="bar-p0" class="pv-bar pv-bar-amber" style="width: 50%;"></div>
                </div>
              </div>
              <div class="pv-prob-row">
                <div class="pv-prob-labels">
                  <span>Class 1 (Positive)</span>
                  <span id="label-p1">50.0%</span>
                </div>
                <div class="pv-bar-track">
                  <div id="bar-p1" class="pv-bar pv-bar-emerald" style="width: 50%;"></div>
                </div>
              </div>
            </div>

            <div class="pv-prediction-verdict" id="prediction-verdict">
              Memproses prediksi...
            </div>
          </div>
        </section>

        <!-- TAB 3: VE HTTP FRAMEWORK -->
        <section id="tab-ve" class="pv-panel">
          <div class="pv-card">
            <div class="pv-card-header">
              <h3>In-Memory Dispatcher & Router Simulator</h3>
              <span class="pv-chip pv-chip-purple">Micro-HTTP Engine</span>
            </div>
            <p class="pv-desc">Uji coba simulasi request HTTP internal langsung ke pipeline Router dan Dispatcher Ve tanpa butuh server eksternal.</p>

            <div class="pv-actions">
              <button class="pv-btn pv-btn-primary pv-btn-route" data-method="GET" data-url="/health">GET /health</button>
              <button class="pv-btn pv-btn-primary pv-btn-route" data-method="GET" data-url="/model">GET /model</button>
              <button class="pv-btn pv-btn-primary pv-btn-route" data-method="POST" data-url="/predict" data-body='{"features":[0.5,-0.2,0.9]}'>POST /predict</button>
              <button class="pv-btn pv-btn-secondary pv-btn-route" data-method="GET" data-url="/unknown-route">GET /404-test</button>
            </div>

            <div class="pv-result-box" style="margin-top: 16px;">
              <div class="pv-res-title">Simulasi Respon HTTP Server:</div>
              <div id="res-ve-dispatch" class="pv-code-block">Klik salah satu route di atas untuk mengirim virtual request.</div>
            </div>
          </div>
        </section>

        <!-- TAB 4: DAG ARCHITECTURE -->
        <section id="tab-dag" class="pv-panel">
          <div class="pv-card">
            <div class="pv-card-header">
              <h3>7-Layer Topological Dependency Hierarchy</h3>
              <span class="pv-chip pv-chip-cyan">Clean Architecture</span>
            </div>
            <p class="pv-desc">Visualisasi arsitektur paralel DAG dari layer 0 (leaf modules) hingga layer 7 (entry points) dengan 0 siklus circular dependency.</p>

            <div class="pv-layers-list">
              <div class="pv-layer-item">
                <div class="pv-layer-badge">Layer 0 (Leaf Modules)</div>
                <div class="pv-layer-files">
                  <span class="pv-tag">dtype.js</span>
                  <span class="pv-tag">device.js</span>
                  <span class="pv-tag">config.js</span>
                  <span class="pv-tag">rng.js</span>
                  <span class="pv-tag">compare.js</span>
                  <span class="pv-tag">EventEmitter.js</span>
                  <span class="pv-tag">HttpError.js</span>
                  <span class="pv-tag">httpStatus.js</span>
                  <span class="pv-tag">headers.js</span>
                  <span class="pv-tag">middleware.js</span>
                  <span class="pv-tag">parseCookies.js</span>
                  <span class="pv-tag">parseUrl.js</span>
                  <span class="pv-tag">path.js</span>
                  <span class="pv-tag">router.js</span>
                  <span class="pv-tag">signals.js</span>
                </div>
              </div>

              <div class="pv-layer-item">
                <div class="pv-layer-badge">Layer 1 (Core Handles & IO)</div>
                <div class="pv-layer-files">
                  <span class="pv-tag">handle.js</span>
                  <span class="pv-tag">parseBody.js</span>
                  <span class="pv-tag">respond.js</span>
                  <span class="pv-tag">Server.js</span>
                </div>
              </div>

              <div class="pv-layer-item">
                <div class="pv-layer-badge">Layer 2 (Operators)</div>
                <div class="pv-layer-files">
                  <span class="pv-tag">factory.js</span>
                  <span class="pv-tag">elementwise.js</span>
                  <span class="pv-tag">linalg.js</span>
                  <span class="pv-tag">reductions.js</span>
                  <span class="pv-tag">shape.js</span>
                  <span class="pv-tag">cast.js</span>
                  <span class="pv-tag">indexing.js</span>
                  <span class="pv-tag">gather.js</span>
                  <span class="pv-tag">random.js</span>
                  <span class="pv-tag">staticFile.js</span>
                </div>
              </div>

              <div class="pv-layer-item">
                <div class="pv-layer-badge">Layer 3 & 4 (Framework Facades)</div>
                <div class="pv-layer-files">
                  <span class="pv-tag">manipulation.js</span>
                  <span class="pv-tag">Dispatcher.js</span>
                  <span class="pv-tag pv-tag-accent">Vessert.js</span>
                  <span class="pv-tag pv-tag-accent">Application.js</span>
                </div>
              </div>

              <div class="pv-layer-item">
                <div class="pv-layer-badge">Layer 5, 6 & 7 (App & Entry Points)</div>
                <div class="pv-layer-files">
                  <span class="pv-tag">ve.js</span>
                  <span class="pv-tag">model.js</span>
                  <span class="pv-tag">routes.js</span>
                  <span class="pv-tag pv-tag-gold">src/index.js (Package Entry)</span>
                  <span class="pv-tag pv-tag-gold">src/app/bootstrap.js (Daemon Entry)</span>
                  <span class="pv-tag pv-tag-gold">src/Preview.js (UI Entry)</span>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <!-- FOOTER -->
      <footer class="pv-footer">
        <span>Vessert & Ve Architecture Preview</span>
        <span>•</span>
        <span>Run <code>npm test</code> for pure CLI tests</span>
        <span>•</span>
        <span>Run <code>npm run audit</code> for full DAG tracing</span>
      </footer>
    </div>
  `;

  // Inject Stylesheet
  injectPreviewStyles();

  // Attach Interaction Handlers
  setupTabs();
  setupMatmul();
  setupFactory();
  setupModelInference();
  setupVeRouter();
}

/**
 * Tab Navigation Logic
 */
function setupTabs() {
  const tabs = document.querySelectorAll('.pv-tab');
  const panels = document.querySelectorAll('.pv-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      tabs.forEach(t => t.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) targetPanel.classList.add('active');
    });
  });
}

/**
 * Matmul Interactive Logic
 */
function setupMatmul() {
  const btnCalc = document.getElementById('btn-calc-matmul');
  const btnRand = document.getElementById('btn-rand-matmul');
  const resBox = document.getElementById('res-matmul');

  function calculate() {
    try {
      const a00 = parseFloat(document.getElementById('ma00').value) || 0;
      const a01 = parseFloat(document.getElementById('ma01').value) || 0;
      const a10 = parseFloat(document.getElementById('ma10').value) || 0;
      const a11 = parseFloat(document.getElementById('ma11').value) || 0;

      const b00 = parseFloat(document.getElementById('mb00').value) || 0;
      const b01 = parseFloat(document.getElementById('mb01').value) || 0;
      const b10 = parseFloat(document.getElementById('mb10').value) || 0;
      const b11 = parseFloat(document.getElementById('mb11').value) || 0;

      const a = Vessert.from([a00, a01, a10, a11], [2, 2]);
      const b = Vessert.from([b00, b01, b10, b11], [2, 2]);
      const c = a.matmul(b);
      const arr = c.toArray();

      resBox.innerHTML = `[Matrix C (2x2)]:
  ┌ ${arr[0].toFixed(2).padStart(8)}, ${arr[1].toFixed(2).padStart(8)} ┐
  └ ${arr[2].toFixed(2).padStart(8)}, ${arr[3].toFixed(2).padStart(8)} ┘
Shape: [${c.shape.join(', ')}] | Dtype: ${c.dtype} | Total Sum: ${c.sum().item().toFixed(2)}`;
    } catch (err) {
      resBox.textContent = `Error: ${err.message}`;
    }
  }

  if (btnCalc) btnCalc.addEventListener('click', calculate);
  if (btnRand) {
    btnRand.addEventListener('click', () => {
      ['ma00', 'ma01', 'ma10', 'ma11', 'mb00', 'mb01', 'mb10', 'mb11'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = (Math.floor(Math.random() * 9) - 4).toString();
      });
      calculate();
    });
  }

  calculate();
}

/**
 * Factory and reductions playground
 */
function setupFactory() {
  const box = document.getElementById('res-factory');

  function showTensor(label, t) {
    box.innerHTML = `Generated ${label}:
Data:  [${t.toArray().slice(0, 16).map(n => typeof n === 'number' ? n.toFixed(2) : n).join(', ')}${t.size > 16 ? '...' : ''}]
Shape: [${t.shape.join(', ')}]
Dtype: ${t.dtype}
Size:  ${t.size} elements`;
  }

  document.getElementById('btn-tensor-zeros')?.addEventListener('click', () => {
    showTensor('Vessert.zeros([3, 3])', Vessert.zeros([3, 3]));
  });
  document.getElementById('btn-tensor-ones')?.addEventListener('click', () => {
    showTensor('Vessert.ones([2, 4])', Vessert.ones([2, 4]));
  });
  document.getElementById('btn-tensor-eye')?.addEventListener('click', () => {
    showTensor('Vessert.eye(4)', Vessert.eye(4));
  });
  document.getElementById('btn-tensor-arange')?.addEventListener('click', () => {
    showTensor('Vessert.arange(0, 10, 2)', Vessert.arange(0, 10, 2));
  });
  document.getElementById('btn-tensor-randn')?.addEventListener('click', () => {
    showTensor('Vessert.randn([3, 3])', Vessert.randn([3, 3]));
  });
}

/**
 * Model inference live playground
 */
function setupModelInference() {
  const s1 = document.getElementById('slider-f1');
  const s2 = document.getElementById('slider-f2');
  const s3 = document.getElementById('slider-f3');
  const v1 = document.getElementById('val-f1');
  const v2 = document.getElementById('val-f2');
  const v3 = document.getElementById('val-f3');
  const bar0 = document.getElementById('bar-p0');
  const bar1 = document.getElementById('bar-p1');
  const lbl0 = document.getElementById('label-p0');
  const lbl1 = document.getElementById('label-p1');
  const verdict = document.getElementById('prediction-verdict');

  function update() {
    if (!s1 || !s2 || !s3) return;
    const f1 = parseFloat(s1.value);
    const f2 = parseFloat(s2.value);
    const f3 = parseFloat(s3.value);

    v1.textContent = f1.toFixed(2);
    v2.textContent = f2.toFixed(2);
    v3.textContent = f3.toFixed(2);

    try {
      const pred = predict([f1, f2, f3]);
      const probs = pred.toArray();
      const p0 = Math.max(0, Math.min(1, probs[0]));
      const p1 = Math.max(0, Math.min(1, probs[1]));

      const p0Pct = (p0 * 100).toFixed(1);
      const p1Pct = (p1 * 100).toFixed(1);

      bar0.style.width = `${p0Pct}%`;
      bar1.style.width = `${p1Pct}%`;
      lbl0.textContent = `${p0Pct}%`;
      lbl1.textContent = `${p1Pct}%`;

      const bestClass = p1 > p0 ? 1 : 0;
      const confidence = (Math.max(p0, p1) * 100).toFixed(1);

      verdict.innerHTML = `Hasil Prediksi: <strong class="${bestClass === 1 ? 'pv-text-green' : 'pv-text-amber'}">Class ${bestClass} (${bestClass === 1 ? 'Positive' : 'Negative'})</strong> dengan keyakinan <strong>${confidence}%</strong>.`;
    } catch (e) {
      verdict.textContent = 'Error inference: ' + e.message;
    }
  }

  [s1, s2, s3].forEach(s => s && s.addEventListener('input', update));
  update();
}

/**
 * Ve Router Virtual Simulation
 */
function setupVeRouter() {
  const buttons = document.querySelectorAll('.pv-btn-route');
  const resBox = document.getElementById('res-ve-dispatch');

  // Build an in-memory Application instance
  const app = new ve.Application();
  mountRoutes(app);

  buttons.forEach(btn => {
    btn.addEventListener('click', async () => {
      const method = btn.getAttribute('data-method');
      const url = btn.getAttribute('data-url');
      const bodyStr = btn.getAttribute('data-body');

      resBox.textContent = `Dispatching ${method} ${url}...`;

      try {
        const start = performance.now();
        // Simulate an HTTP req/res in memory
        let statusCode = 200;
        let responseHeaders = {};
        let responseBody = '';

        const fakeReq = {
          method,
          url,
          headers: {
            'content-type': 'application/json',
            'host': 'localhost:3000'
          },
          on(evt, handler) {
            if (evt === 'data' && bodyStr) {
              handler(typeof Buffer !== 'undefined' && typeof Buffer.from === 'function' ? Buffer.from(bodyStr) : bodyStr);
            }
            if (evt === 'end') {
              setTimeout(handler, 2);
            }
            return this;
          }
        };

        const fakeRes = {
          writeHead(code, headers) {
            statusCode = code;
            responseHeaders = headers || {};
          },
          end(chunk) {
            responseBody = chunk || '';
          }
        };

        // Match with router
        const match = app._router.match(method, url);
        if (!match || !match.handler) {
          statusCode = 404;
          responseBody = JSON.stringify({ error: 'Not Found', path: url });
        } else {
          // Direct execute route handler
          await match.handler(fakeReq, fakeRes, match.params);
        }

        const elapsed = (performance.now() - start).toFixed(2);
        resBox.innerHTML = `HTTP Status: <strong class="${statusCode === 200 ? 'pv-text-green' : 'pv-text-amber'}">${statusCode}</strong> (Latensi: ${elapsed}ms)
Header:
  Content-Type: ${responseHeaders['Content-Type'] || responseHeaders['content-type'] || 'application/json'}
Body:
${responseBody ? JSON.stringify(typeof responseBody === 'string' ? JSON.parse(responseBody) : responseBody, null, 2) : '(empty)'}`;
      } catch (err) {
        resBox.textContent = `Virtual Server Error: ${err.message}`;
      }
    });
  });
}

/**
 * CSS Styling Injection
 */
function injectPreviewStyles() {
  if (document.getElementById('pv-styles')) return;
  const style = document.createElement('style');
  style.id = 'pv-styles';
  style.textContent = `
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 0;
      background: #090d16;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    .pv-root {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      min-height: 100vh;
    }
    .pv-header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid #1e293b;
      gap: 16px;
    }
    .pv-logo {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .pv-logo-symbol {
      font-size: 28px;
      color: #38bdf8;
      background: #0f172a;
      border: 1px solid #1e293b;
      width: 46px;
      height: 46px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 10px;
    }
    .pv-title {
      font-size: 20px;
      font-weight: 700;
      margin: 0;
      color: #f8fafc;
      letter-spacing: -0.02em;
    }
    .pv-subtitle {
      font-size: 13px;
      margin: 4px 0 0 0;
      color: #94a3b8;
    }
    .pv-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .pv-badge {
      background: #1e293b;
      color: #cbd5e1;
      padding: 6px 12px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 500;
      border: 1px solid #334155;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }
    .pv-badge-green {
      background: rgba(34, 197, 94, 0.1);
      color: #4ade80;
      border-color: rgba(34, 197, 94, 0.3);
    }
    .pv-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 8px #22c55e;
    }
    .pv-nav {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid #1e293b;
      padding-bottom: 12px;
      overflow-x: auto;
    }
    .pv-tab {
      background: transparent;
      border: 1px solid transparent;
      color: #94a3b8;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      transition: all 0.15s ease;
      white-space: nowrap;
    }
    .pv-tab:hover {
      color: #f1f5f9;
      background: #1e293b;
    }
    .pv-tab.active {
      background: #1e293b;
      color: #38bdf8;
      border-color: #334155;
      font-weight: 600;
    }
    .pv-main {
      flex: 1;
    }
    .pv-panel {
      display: none;
    }
    .pv-panel.active {
      display: block;
      animation: fadeIn 0.2s ease-in-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .pv-grid-2 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 20px;
    }
    .pv-card {
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .pv-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }
    .pv-card-header h3 {
      font-size: 16px;
      margin: 0;
      color: #f8fafc;
      font-weight: 600;
    }
    .pv-chip {
      font-size: 11px;
      padding: 3px 8px;
      background: #1e293b;
      color: #94a3b8;
      border-radius: 6px;
      border: 1px solid #334155;
      white-space: nowrap;
    }
    .pv-chip-blue { background: rgba(56, 189, 248, 0.1); color: #38bdf8; border-color: rgba(56, 189, 248, 0.3); }
    .pv-chip-purple { background: rgba(168, 85, 247, 0.1); color: #c084fc; border-color: rgba(168, 85, 247, 0.3); }
    .pv-chip-cyan { background: rgba(6, 182, 212, 0.1); color: #22d3ee; border-color: rgba(6, 182, 212, 0.3); }
    .pv-desc {
      font-size: 13px;
      color: #94a3b8;
      margin: 0;
      line-height: 1.5;
    }
    .pv-matrix-grid {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      justify-content: center;
    }
    .pv-matrix-box label {
      display: block;
      font-size: 11px;
      color: #64748b;
      margin-bottom: 6px;
      text-transform: uppercase;
      font-weight: 600;
    }
    .pv-inputs-2x2 {
      display: grid;
      grid-template-columns: repeat(2, 60px);
      gap: 6px;
    }
    .pv-input {
      background: #1e293b;
      border: 1px solid #334155;
      color: #f8fafc;
      font-size: 14px;
      padding: 8px;
      border-radius: 6px;
      text-align: center;
      font-weight: 600;
      width: 100%;
    }
    .pv-input:focus {
      outline: none;
      border-color: #38bdf8;
    }
    .pv-matrix-op {
      font-size: 18px;
      color: #64748b;
      font-weight: bold;
    }
    .pv-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .pv-btn {
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
      border: 1px solid transparent;
      white-space: nowrap;
    }
    .pv-btn-primary {
      background: #0284c7;
      color: #ffffff;
    }
    .pv-btn-primary:hover {
      background: #0369a1;
    }
    .pv-btn-secondary {
      background: #1e293b;
      color: #e2e8f0;
      border-color: #334155;
    }
    .pv-btn-secondary:hover {
      background: #334155;
    }
    .pv-btn-outline {
      background: transparent;
      color: #94a3b8;
      border-color: #334155;
    }
    .pv-btn-outline:hover {
      background: #1e293b;
      color: #f8fafc;
    }
    .pv-result-box {
      background: #030712;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 12px;
    }
    .pv-res-title {
      font-size: 12px;
      color: #64748b;
      margin-bottom: 6px;
      font-weight: 600;
    }
    .pv-code-block {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      color: #38bdf8;
      white-space: pre-wrap;
      line-height: 1.5;
    }
    .pv-model-controls {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .pv-slider-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .pv-slider-header {
      display: flex;
      justify-content: space-between;
      font-size: 13px;
      color: #cbd5e1;
    }
    .pv-val-num {
      font-family: ui-monospace, monospace;
      font-weight: 600;
      color: #38bdf8;
    }
    .pv-range {
      width: 100%;
      accent-color: #38bdf8;
    }
    .pv-prob-container {
      margin-top: 8px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .pv-prob-container h4 {
      margin: 0;
      font-size: 13px;
      color: #cbd5e1;
    }
    .pv-prob-row {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .pv-prob-labels {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: #94a3b8;
    }
    .pv-bar-track {
      height: 8px;
      background: #1e293b;
      border-radius: 4px;
      overflow: hidden;
    }
    .pv-bar {
      height: 100%;
      border-radius: 4px;
      transition: width 0.2s ease-out;
    }
    .pv-bar-emerald { background: #10b981; }
    .pv-bar-amber { background: #f59e0b; }
    .pv-prediction-verdict {
      padding: 12px;
      background: #030712;
      border-radius: 8px;
      border: 1px solid #1e293b;
      font-size: 14px;
      color: #cbd5e1;
    }
    .pv-text-green { color: #4ade80; }
    .pv-text-amber { color: #fbbf24; }
    .pv-layers-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .pv-layer-item {
      background: #030712;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .pv-layer-badge {
      font-size: 12px;
      font-weight: 700;
      color: #38bdf8;
    }
    .pv-layer-files {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .pv-tag {
      font-size: 11px;
      background: #1e293b;
      color: #cbd5e1;
      padding: 2px 8px;
      border-radius: 4px;
      font-family: ui-monospace, monospace;
      border: 1px solid #334155;
    }
    .pv-tag-accent {
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border-color: rgba(56, 189, 248, 0.4);
    }
    .pv-tag-gold {
      background: rgba(251, 191, 36, 0.15);
      color: #fbbf24;
      border-color: rgba(251, 191, 36, 0.4);
    }
    .pv-footer {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 8px;
      font-size: 12px;
      color: #64748b;
      padding-top: 16px;
      border-top: 1px solid #1e293b;
    }
    .pv-footer code {
      background: #1e293b;
      color: #38bdf8;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 11px;
    }
  `;
  document.head.appendChild(style);
}

// Auto-mount if running in browser DOM
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => renderPreview());
  } else {
    renderPreview();
  }
}
