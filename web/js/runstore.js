/*
 * Runs, kept in the browser.
 *
 * IndexedDB rather than localStorage, which holds about five megabytes of text
 * and would not survive a single frame. Frames go in one at a time as they are
 * produced, so a run that is still going is already saved, and closing the tab
 * mid-run loses only the iteration in flight.
 *
 * This also takes the frames out of Python's hands entirely. They used to sit
 * in the worker's memory for the life of the page, which is what put a ceiling
 * on how long a run could go. Now the engine produces a frame, it is written
 * here, and the interpreter forgets it.
 *
 * Loaded both by the page and by the worker — importScripts and a plain script
 * tag both work on a file with no imports of its own.
 */

const RunStore = {
  DB: 'graphoflife',
  VERSION: 1,
  _db: null,

  /**
   * Object stores:
   *   runs        one record per run, its metadata
   *   frames      one record per frame, keyed [runId, index]
   *   checkpoints one blob per run, enough to carry on from
   */
  open() {
    if (this._db) return Promise.resolve(this._db);
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.DB, this.VERSION);

      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains('runs')) {
          db.createObjectStore('runs', { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains('frames')) {
          // Compound key, so every frame of a run is one contiguous range and
          // deleting a run is a single ranged delete rather than a scan.
          db.createObjectStore('frames', { keyPath: ['runId', 'index'] });
        }
        if (!db.objectStoreNames.contains('checkpoints')) {
          db.createObjectStore('checkpoints', { keyPath: 'runId' });
        }
      };

      request.onsuccess = () => { this._db = request.result; resolve(this._db); };
      request.onerror = () => reject(request.error || new Error('IndexedDB refused to open'));
    });
  },

  async _tx(stores, mode, run) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(stores, mode);
      let result;
      tx.oncomplete = () => resolve(result);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error || new Error('the write was aborted, most likely out of space'));
      result = run(tx);
    });
  },

  _await(request) {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  },

  // ---- runs ------------------------------------------------------------

  async putRun(meta) {
    await this._tx(['runs'], 'readwrite', tx => tx.objectStore('runs').put(meta));
    return meta;
  },

  async getRun(runId) {
    const db = await this.open();
    return this._await(db.transaction('runs').objectStore('runs').get(runId));
  },

  async listRuns() {
    const db = await this.open();
    const all = await this._await(db.transaction('runs').objectStore('runs').getAll());
    return all.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  },

  async deleteRun(runId) {
    await this._tx(['runs', 'frames', 'checkpoints'], 'readwrite', tx => {
      tx.objectStore('runs').delete(runId);
      tx.objectStore('checkpoints').delete(runId);
      // Everything from [id, 0] up to [id, ∞): the whole run in one range.
      tx.objectStore('frames').delete(
        IDBKeyRange.bound([runId, -Infinity], [runId, Infinity])
      );
    });
  },

  // ---- frames ----------------------------------------------------------

  /** Written as one transaction, so a slice either lands or does not. */
  async putFrames(runId, startIndex, frames) {
    if (!frames.length) return;
    await this._tx(['frames'], 'readwrite', tx => {
      const store = tx.objectStore('frames');
      frames.forEach((frame, offset) => {
        store.put({ runId, index: startIndex + offset, frame });
      });
    });
  },

  async getFrame(runId, index) {
    const db = await this.open();
    const row = await this._await(
      db.transaction('frames').objectStore('frames').get([runId, index])
    );
    if (!row) throw new Error(`frame ${index} of ${runId} is not stored`);
    return row.frame;
  },

  /** Frames at a fixed stride, for the charts. Read in one pass. */
  async getFramesStrided(runId, stride) {
    const db = await this.open();
    const store = db.transaction('frames').objectStore('frames');
    const range = IDBKeyRange.bound([runId, -Infinity], [runId, Infinity]);
    const wanted = [];

    await new Promise((resolve, reject) => {
      const cursor = store.openCursor(range);
      cursor.onsuccess = () => {
        const at = cursor.result;
        if (!at) { resolve(); return; }
        // Both phases of a sampled iteration are kept, so a phase filter still
        // has game frames to show.
        if (Math.floor(at.value.index / 2) % stride === 0) wanted.push(at.value.frame);
        at.continue();
      };
      cursor.onerror = () => reject(cursor.error);
    });
    return wanted;
  },

  // ---- checkpoints -----------------------------------------------------

  async putCheckpoint(runId, bytes) {
    await this._tx(['checkpoints'], 'readwrite',
                   tx => tx.objectStore('checkpoints').put({ runId, bytes }));
  },

  async getCheckpoint(runId) {
    const db = await this.open();
    const row = await this._await(
      db.transaction('checkpoints').objectStore('checkpoints').get(runId)
    );
    return row ? row.bytes : null;
  },

  // ---- housekeeping ----------------------------------------------------

  /**
   * Ask the browser not to evict this data.
   *
   * Without it, storage is "best effort" and may be cleared when the disk gets
   * tight — which for a run that took an hour is not a detail. Refusal is
   * normal and not worth complaining about; the data is still written.
   */
  async requestPersistence() {
    if (!navigator.storage || !navigator.storage.persist) return false;
    try {
      return await navigator.storage.persisted() || await navigator.storage.persist();
    } catch (err) {
      return false;
    }
  },

  async usage() {
    if (!navigator.storage || !navigator.storage.estimate) return null;
    try {
      const { usage, quota } = await navigator.storage.estimate();
      return { usage, quota };
    } catch (err) {
      return null;
    }
  }
};

if (typeof self !== 'undefined' && typeof window === 'undefined') {
  self.RunStore = RunStore;   // inside the worker
}
