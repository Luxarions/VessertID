/**
 * headers.js
 * Changes when: the security header policy changes.
 * @memberof ve
 */

/**
 * @memberof ve
 * @type {Readonly<Record<string,string>>}
 */
const SECURITY_HEADERS = Object.freeze({
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'Referrer-Policy': 'no-referrer',
  'Cross-Origin-Resource-Policy': 'same-origin',
});

/**
 * @memberof ve
 * @param {string} name
 * @param {string} value
 */
function assertHeaderSafe(name, value) {
  const v = String(value);
  if (/[\r\n]/.test(name) || /[\r\n]/.test(v)) {
    throw new Error(`header: illegal CR/LF in ${name}`);
  }
}

/**
 * @memberof ve
 * @param {Record<string,string>} [extra]
 * @returns {Record<string,string>}
 */
function withSecurityHeaders(extra = {}) {
  return { ...SECURITY_HEADERS, ...extra };
}

export { SECURITY_HEADERS, assertHeaderSafe, withSecurityHeaders };
