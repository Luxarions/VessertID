/**
 * src/Preview.js — Pure JavaScript Logic & Controller
 * No raw HTML templates or inline CSS injection.
 * Connects the UI elements defined in index.html to the underlying
 * Vessert tensor engine, Ve micro-framework, and neural inference model.
 */

import { Vessert, Dtype, Device, config, ve } from './index.js';
import { predict, info } from './app/model.js';
import { mountRoutes } from './app/routes.js';

/**
 * Pure JS calculation for 2x2 Matrix Multiplication.
 * @param {number[]} a - 4-element array for Matrix A
 * @param {number[]} b - 4-element array for Matrix B
 * @returns {{ result: number[], shape: number[], dtype: string, sum: number }}
 */
export function computeMatmul(a, b) {
  const tensorA = Vessert.from(a, [2, 2]);
  const tensorB = Vessert.from(b, [2, 2]);
  const tensorC = tensorA.matmul(tensorB);
  return {
    result: tensorC.toArray(),
    shape: tensorC.shape,
    dtype: tensorC.dtype,
    sum: tensorC.sum().item(),
  };
}

/**
 * Pure JS tensor factory generator.
 * @param {'zeros' | 'ones' | 'eye' | 'arange' | 'randn'} type
 * @returns {{ label: string, data: any[], shape: number[], dtype: string, size: number }}
 */
export function generateTensorSample(type) {
  let tensor;
  let label = '';
  switch (type) {
    case 'zeros':
      tensor = Vessert.zeros([3, 3]);
      label = 'Vessert.zeros([3, 3])';
      break;
    case 'ones':
      tensor = Vessert.ones([2, 4]);
      label = 'Vessert.ones([2, 4])';
      break;
    case 'eye':
      tensor = Vessert.eye(4);
      label = 'Vessert.eye(4)';
      break;
    case 'arange':
      tensor = Vessert.arange(0, 10, 2);
      label = 'Vessert.arange(0, 10, 2)';
      break;
    case 'randn':
      tensor = Vessert.randn([3, 3]);
      label = 'Vessert.randn([3, 3])';
      break;
    default:
      tensor = Vessert.zeros([2, 2]);
      label = 'Vessert.zeros([2, 2])';
  }

  return {
    label,
    data: tensor.toArray(),
    shape: tensor.shape,
    dtype: tensor.dtype,
    size: tensor.size,
  };
}

/**
 * Pure JS model inference runner.
 * @param {number} f1
 * @param {number} f2
 * @param {number} f3
 * @returns {{ p0: number, p1: number, bestClass: number, confidence: number }}
 */
export function runInference(f1, f2, f3) {
  const pred = predict([f1, f2, f3]);
  const probs = pred.toArray();
  const p0 = Math.max(0, Math.min(1, probs[0]));
  const p1 = Math.max(0, Math.min(1, probs[1]));
  const bestClass = p1 > p0 ? 1 : 0;
  const confidence = Math.max(p0, p1);

  return { p0, p1, bestClass, confidence };
}

/**
 * Pure JS in-memory Ve virtual request dispatcher.
 * @param {string} method
 * @param {string} url
 * @param {string} [bodyStr]
 * @returns {Promise<{ status: number, latencyMs: number, contentType: string, data: any }>}
 */
export async function dispatchVirtualRoute(method, url, bodyStr = '') {
  const app = new ve.Application();
  mountRoutes(app);

  const start = performance.now();
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

  const match = app._router.match(method, url);
  if (!match || !match.handler) {
    statusCode = 404;
    responseBody = JSON.stringify({ error: 'Not Found', path: url });
  } else {
    await match.handler(fakeReq, fakeRes, match.params);
  }

  const latencyMs = parseFloat((performance.now() - start).toFixed(2));
  let parsedData;
  try {
    parsedData = typeof responseBody === 'string' ? JSON.parse(responseBody) : responseBody;
  } catch {
    parsedData = responseBody;
  }

  return {
    status: statusCode,
    latencyMs,
    contentType: responseHeaders['Content-Type'] || responseHeaders['content-type'] || 'application/json',
    data: parsedData,
  };
}

/**
 * Binds and initializes the DOM elements present in index.html.
 * Pure JS DOM event binding — does not create HTML strings.
 */
export function initPreview() {
  if (typeof document === 'undefined') return;

  // 1. Tab Navigation
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

  // 2. Matmul Calculator
  const btnCalc = document.getElementById('btn-calc-matmul');
  const btnRand = document.getElementById('btn-rand-matmul');
  const resMatmul = document.getElementById('res-matmul');

  function updateMatmul() {
    if (!resMatmul) return;
    try {
      const a = [
        parseFloat(document.getElementById('ma00')?.value) || 0,
        parseFloat(document.getElementById('ma01')?.value) || 0,
        parseFloat(document.getElementById('ma10')?.value) || 0,
        parseFloat(document.getElementById('ma11')?.value) || 0,
      ];
      const b = [
        parseFloat(document.getElementById('mb00')?.value) || 0,
        parseFloat(document.getElementById('mb01')?.value) || 0,
        parseFloat(document.getElementById('mb10')?.value) || 0,
        parseFloat(document.getElementById('mb11')?.value) || 0,
      ];

      const out = computeMatmul(a, b);
      const arr = out.result;
      resMatmul.textContent = `[Matrix C (2x2)]:
  ┌ ${arr[0].toFixed(2).padStart(8)}, ${arr[1].toFixed(2).padStart(8)} ┐
  └ ${arr[2].toFixed(2).padStart(8)}, ${arr[3].toFixed(2).padStart(8)} ┘
Shape: [${out.shape.join(', ')}] | Dtype: ${out.dtype} | Total Sum: ${out.sum.toFixed(2)}`;
    } catch (err) {
      resMatmul.textContent = `Error: ${err.message}`;
    }
  }

  btnCalc?.addEventListener('click', updateMatmul);
  btnRand?.addEventListener('click', () => {
    ['ma00', 'ma01', 'ma10', 'ma11', 'mb00', 'mb01', 'mb10', 'mb11'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.value = (Math.floor(Math.random() * 9) - 4).toString();
    });
    updateMatmul();
  });
  updateMatmul();

  // 3. Tensor Factory Sample Buttons
  const resFactory = document.getElementById('res-factory');
  const factoryTypes = ['zeros', 'ones', 'eye', 'arange', 'randn'];
  factoryTypes.forEach(t => {
    const btn = document.getElementById(`btn-tensor-${t}`);
    btn?.addEventListener('click', () => {
      const sample = generateTensorSample(t);
      if (resFactory) {
        resFactory.textContent = `Generated ${sample.label}:
Data:  [${sample.data.slice(0, 16).map(n => typeof n === 'number' ? n.toFixed(2) : n).join(', ')}${sample.size > 16 ? '...' : ''}]
Shape: [${sample.shape.join(', ')}]
Dtype: ${sample.dtype}
Size:  ${sample.size} elements`;
      }
    });
  });

  // 4. Model Inference Sliders
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

  function updateInference() {
    if (!s1 || !s2 || !s3) return;
    const f1 = parseFloat(s1.value);
    const f2 = parseFloat(s2.value);
    const f3 = parseFloat(s3.value);

    if (v1) v1.textContent = f1.toFixed(2);
    if (v2) v2.textContent = f2.toFixed(2);
    if (v3) v3.textContent = f3.toFixed(2);

    try {
      const res = runInference(f1, f2, f3);
      const p0Pct = (res.p0 * 100).toFixed(1);
      const p1Pct = (res.p1 * 100).toFixed(1);

      if (bar0) bar0.style.width = `${p0Pct}%`;
      if (bar1) bar1.style.width = `${p1Pct}%`;
      if (lbl0) lbl0.textContent = `${p0Pct}%`;
      if (lbl1) lbl1.textContent = `${p1Pct}%`;

      if (verdict) {
        const clsColor = res.bestClass === 1 ? 'pv-text-green' : 'pv-text-amber';
        const clsName = res.bestClass === 1 ? 'Positive' : 'Negative';
        verdict.innerHTML = `Hasil Prediksi: <strong class="${clsColor}">Class ${res.bestClass} (${clsName})</strong> dengan keyakinan <strong>${(res.confidence * 100).toFixed(1)}%</strong>.`;
      }
    } catch (err) {
      if (verdict) verdict.textContent = 'Error inference: ' + err.message;
    }
  }

  [s1, s2, s3].forEach(s => s?.addEventListener('input', updateInference));
  updateInference();

  // 5. Ve Router Buttons
  const routeButtons = document.querySelectorAll('.pv-btn-route');
  const resVe = document.getElementById('res-ve-dispatch');
  routeButtons.forEach(btn => {
    btn.addEventListener('click', async () => {
      const method = btn.getAttribute('data-method') || 'GET';
      const url = btn.getAttribute('data-url') || '/';
      const body = btn.getAttribute('data-body') || '';

      if (resVe) resVe.textContent = `Dispatching ${method} ${url}...`;
      try {
        const out = await dispatchVirtualRoute(method, url, body);
        if (resVe) {
          const statusClass = out.status === 200 ? 'pv-text-green' : 'pv-text-amber';
          resVe.innerHTML = `HTTP Status: <strong class="${statusClass}">${out.status}</strong> (Latensi: ${out.latencyMs}ms)
Header:
  Content-Type: ${out.contentType}
Body:
${JSON.stringify(out.data, null, 2)}`;
        }
      } catch (err) {
        if (resVe) resVe.textContent = `Virtual Server Error: ${err.message}`;
      }
    });
  });
}

/**
 * Backward compatibility alias for renderPreview
 */
export function renderPreview() {
  initPreview();
}

export default {
  initPreview,
  renderPreview,
  computeMatmul,
  generateTensorSample,
  runInference,
  dispatchVirtualRoute,
};
