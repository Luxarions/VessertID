/**
 * path.js
 * Changes when: path safety policy (traversal, symlink, encoding) changes.
 * @memberof ve
 */

import fs from 'node:fs/promises';
import path from 'node:path';

/**
 * Lexical-only resolution. Does NOT touch the filesystem.
 * @memberof ve
 * @param {string} rootDir
 * @param {string} requestUrl
 * @param {string} [indexFallback]
 * @returns {string|null}
 */
function resolveSafePath(rootDir, requestUrl, indexFallback = 'index.html') {
  const normalizedRoot = path.resolve(rootDir);
  const rootWithSep = normalizedRoot.endsWith(path.sep) ? normalizedRoot : normalizedRoot + path.sep;
  const raw = String(requestUrl).split('?')[0].split('#')[0];

  let decoded;
  try { decoded = decodeURIComponent(raw); } catch { return null; }
  if (decoded.includes('\0')) return null;

  let rel = decoded.replace(/^\/+/, '');
  if (!rel || rel.endsWith('/')) rel = path.posix.join(rel, indexFallback);

  const resolved = path.resolve(normalizedRoot, rel);
  if (resolved !== normalizedRoot && !resolved.startsWith(rootWithSep)) return null;
  return resolved;
}

/**
 * Symlink-aware verification. Returns null if the real path escapes rootDir.
 * @memberof ve
 * @param {string} rootDir
 * @param {string} absPath
 * @returns {Promise<string|null>}
 */
async function verifyRealPath(rootDir, absPath) {
  try {
    const realRoot = await fs.realpath(rootDir);
    const realFile = await fs.realpath(absPath);
    const sep = realRoot.endsWith(path.sep) ? realRoot : realRoot + path.sep;
    if (realFile !== realRoot && !realFile.startsWith(sep)) return null;
    return realFile;
  } catch {
    return null;
  }
}

export { resolveSafePath, verifyRealPath };
