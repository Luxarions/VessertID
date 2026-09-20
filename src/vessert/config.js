/**
 * config.js
 * Changes when: the global library configuration shape changes.
 * @memberof Vessert
 *
 * Deliberately NOT frozen. Mutations flow exclusively through the setter
 * methods on the Vessert class (setBackend, setDevice, setDefaultDtype,
 * setGradEnabled). Freezing this object would make those setters throw in
 * strict mode (ES modules are strict) — a bug that silently disables the
 * entire configuration API.
 */

/**
 * @memberof Vessert
 * @type {{backend:string, device:string, defaultDtype:string, gradEnabled:boolean}}
 */
const config = {
  backend: 'cpu',
  device: 'cpu',
  defaultDtype: 'float32',
  gradEnabled: false,
};

export { config };
