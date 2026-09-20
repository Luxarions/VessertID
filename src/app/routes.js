/**
 * routes.js
 * Changes when: the HTTP API contract changes.
 */

import { ve } from '../ve/ve.js';
import { predict, info } from './model.js';

/**
 * @param {ve.Application} app
 * @returns {ve.Application}
 */
function mountRoutes(app) {
  app.get('/health', (req, res) => ve.sendJson(res, 200, { status: 'ok' }));

  app.get('/model', (req, res) => ve.sendJson(res, 200, info()));

  app.post('/predict', async (req, res) => {
    const body = await ve.readBody(req);
    if (!body || !Array.isArray(body.input)) {
      throw ve.HttpError.badRequest('predict: body.input must be an array');
    }
    ve.sendJson(res, 200, { output: predict(body.input).toArray() });
  });

  return app;
}

export { mountRoutes };
