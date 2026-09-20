/**
 * parseUrl.js
 * Changes when: URL and query-string parsing changes.
 * @memberof ve
 */

/**
 * @memberof ve
 * @param {*} req
 * @returns {{pathname: string, query: Record<string,string>, raw: string}}
 */
function parseUrl(req) {
  const raw = req.url || '/';
  const q = raw.indexOf('?');
  const pathname = q === -1 ? raw : raw.slice(0, q);
  const query = q === -1 ? {} : parseQuery(raw.slice(q + 1));
  return { pathname, query, raw };
}

/**
 * @memberof ve
 * @param {string} str
 * @returns {Record<string,string>}
 */
function parseQuery(str) {
  const out = {};
  if (!str) return out;
  for (const pair of str.split('&')) {
    if (!pair) continue;
    const eq = pair.indexOf('=');
    const k = eq === -1 ? pair : pair.slice(0, eq);
    const v = eq === -1 ? '' : pair.slice(eq + 1);
    try {
      out[decodeURIComponent(k.replace(/\+/g, ' '))] = decodeURIComponent(v.replace(/\+/g, ' '));
    } catch { /* skip malformed pairs */ }
  }
  return out;
}

export { parseUrl, parseQuery };
