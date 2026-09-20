/**
 * HttpError.js
 * Changes when: the HTTP error taxonomy changes.
 * @memberof ve
 */

/**
 * @memberof ve
 * @class
 */
class HttpError extends Error {
  constructor(status, message, options = {}) {
    super(message, options.cause ? { cause: options.cause } : undefined);
    this.name = 'HttpError';
    this.status = status;
    this.expose = options.expose ?? status < 500;
    this.code = options.code;
    this.headers = options.headers || {};
  }

  static badRequest(m, o)           { return new HttpError(400, m, o); }
  static unauthorized(m, o)         { return new HttpError(401, m, o); }
  static forbidden(m, o)            { return new HttpError(403, m, o); }
  static notFound(m, o)             { return new HttpError(404, m, o); }
  static methodNotAllowed(m, o)     { return new HttpError(405, m, o); }
  static payloadTooLarge(m, o)      { return new HttpError(413, m, o); }
  static unsupportedMediaType(m, o) { return new HttpError(415, m, o); }
  static internal(m, o)             { return new HttpError(500, m, o); }
}

/**
 * @memberof ve
 * @param {*} err
 * @returns {HttpError}
 */
function toHttpError(err) {
  if (err instanceof HttpError) return err;
  return new HttpError(500, err && err.message ? err.message : 'internal error', { cause: err });
}

export { HttpError, toHttpError };
