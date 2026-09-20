/**
 * router.js
 * Changes when: route matching semantics (patterns, params, wildcards) change.
 * @memberof ve
 */

/**
 * @memberof ve
 * @param {string} pattern
 * @returns {{regex: RegExp, keys: string[]}}
 */
function compilePattern(pattern) {
  const keys = [];
  const parts = pattern.split('/').filter(Boolean);
  const regexParts = parts.map((p) => {
    if (p.startsWith(':')) { keys.push(p.slice(1)); return '([^/]+)'; }
    if (p === '*') { keys.push('wildcard'); return '(.*)'; }
    return p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  });
  return { regex: new RegExp('^/' + regexParts.join('/') + '/?$'), keys };
}

/**
 * @memberof ve
 * @class
 */
class Router {
  constructor() { this._routes = []; }

  add(method, pattern, handler) {
    if (typeof handler !== 'function') throw new Error('router.add: handler must be a function');
    const { regex, keys } = compilePattern(pattern);
    this._routes.push({ method: method.toUpperCase(), pattern, regex, keys, handler });
    return this;
  }

  get(p, h)    { return this.add('GET', p, h); }
  post(p, h)   { return this.add('POST', p, h); }
  put(p, h)    { return this.add('PUT', p, h); }
  delete(p, h) { return this.add('DELETE', p, h); }

  match(method, pathname) {
    const upper = method.toUpperCase();
    const allowed = new Set();
    for (const r of this._routes) {
      const m = r.regex.exec(pathname);
      if (!m) continue;
      if (r.method !== upper) { allowed.add(r.method); continue; }
      const params = {};
      for (let i = 0; i < r.keys.length; i++) params[r.keys[i]] = m[i + 1];
      return { handler: r.handler, params, pattern: r.pattern };
    }
    return allowed.size ? { allowed: [...allowed] } : null;
  }
}

export { Router, compilePattern };
