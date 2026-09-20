/**
 * bootstrap.js
 * Changes when: application wiring changes.
 */

import { ve } from '../ve/ve.js';
import { mountRoutes } from './routes.js';

/**
 * @param {{rootDir?:string, indexFallback?:string, port?:number, host?:string, requestTimeout?:number, headersTimeout?:number}} [options]
 * @returns {Promise<{app: ve.Application, server: ve.Server}>}
 */
async function bootstrap(options = {}) {
  const app = new ve.Application({
    rootDir: options.rootDir ?? './public',
    indexFallback: options.indexFallback ?? 'index.html',
  });

  app.use((req, res, next) => {
    const start = Date.now();
    res.once('finish', () => {
      process.stdout.write(`${req.method} ${req.url} ${res.statusCode} ${Date.now() - start}ms\n`);
    });
    next();
  });

  app.on('error', (err, req) => {
    process.stderr.write(`app error: ${err.message} ${req ? req.url : ''}\n`);
  });

  mountRoutes(app);

  const server = new ve.Server((req, res) => app.handle(req, res), {
    requestTimeout: options.requestTimeout,
    headersTimeout: options.headersTimeout,
  });
  server.installSignals();

  await server.listen(options.port ?? 3000, options.host ?? '0.0.0.0');
  return { app, server };
}

export { bootstrap };
