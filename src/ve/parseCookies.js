/**
 * parseCookies.js
 * Changes when: Cookie header parsing changes.
 * @memberof ve
 */

/**
 * @memberof ve
 * @param {*} req
 * @returns {Record<string,string>}
 */
function parseCookies(req) {
  const header = req.headers.cookie;
  const out = {};
  if (!header) return out;
  for (const part of header.split(';')) {
    const eq = part.indexOf('=');
    if (eq === -1) continue;
    const k = part.slice(0, eq).trim();
    const v = part.slice(eq + 1).trim();
    if (k) out[k] = v;
  }
  return out;
}

export { parseCookies };
