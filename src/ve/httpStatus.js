/**
 * httpStatus.js
 * Changes when: the HTTP reason-phrase registry changes.
 * @memberof ve
 */

/**
 * @memberof ve
 * @type {Readonly<Record<number,string>>}
 */
const REASON = Object.freeze({
  200: 'OK', 201: 'Created', 204: 'No Content',
  301: 'Moved Permanently', 302: 'Found', 304: 'Not Modified',
  400: 'Bad Request', 401: 'Unauthorized', 403: 'Forbidden',
  404: 'Not Found', 405: 'Method Not Allowed', 409: 'Conflict',
  413: 'Payload Too Large', 415: 'Unsupported Media Type',
  422: 'Unprocessable Entity', 429: 'Too Many Requests',
  500: 'Internal Server Error', 502: 'Bad Gateway', 503: 'Service Unavailable',
});

/**
 * @memberof ve
 * @param {number} status
 * @returns {string}
 */
function reasonOf(status) { return REASON[status] || 'Unknown'; }

export { REASON, reasonOf };
