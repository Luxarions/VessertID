/**
 * EventEmitter.js
 * Changes when: event semantics (on/once/emit/off) change.
 * @memberof ve
 */

/**
 * Minimal synchronous event emitter.
 * @memberof ve
 * @class
 */
class EventEmitter {
  constructor() { this._events = new Map(); }

  on(event, listener) {
    if (typeof listener !== 'function') throw new Error('on: listener must be a function');
    const list = this._events.get(event);
    if (list) list.push(listener); else this._events.set(event, [listener]);
    return this;
  }

  once(event, listener) {
    const self = this;
    function wrapper(...args) { self.off(event, wrapper); listener.apply(self, args); }
    wrapper._original = listener;
    return this.on(event, wrapper);
  }

  prependListener(event, listener) {
    const list = this._events.get(event);
    if (list) list.unshift(listener); else this._events.set(event, [listener]);
    return this;
  }

  off(event, listener) {
    if (listener === undefined) { this._events.delete(event); return this; }
    const list = this._events.get(event);
    if (!list) return this;
    const i = list.findIndex((fn) => fn === listener || fn._original === listener);
    if (i !== -1) list.splice(i, 1);
    if (list.length === 0) this._events.delete(event);
    return this;
  }

  removeAllListeners(event) {
    if (event === undefined) this._events.clear(); else this._events.delete(event);
    return this;
  }

  emit(event, ...args) {
    const list = this._events.get(event);
    if (!list || list.length === 0) {
      if (event === 'error') {
        throw args[0] instanceof Error ? args[0] : new Error(String(args[0] ?? 'unknown error'));
      }
      return false;
    }
    const snap = list.slice();
    for (let i = 0; i < snap.length; i++) snap[i].apply(this, args);
    return true;
  }

  listenerCount(event) { const l = this._events.get(event); return l ? l.length : 0; }
  eventNames() { return [...this._events.keys()]; }
  listeners(event) { const l = this._events.get(event); return l ? l.slice() : []; }
}

export { EventEmitter };
