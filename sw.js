/* FitLife Service Worker */
const CACHE_VERSION = 'v8';
const CACHE_NAME = `fitlife-cache-${CACHE_VERSION}`;
const ASSETS_TO_PRECACHE = [
  './',
  './index.html',
  './privacy.html',
  './terms.html',
  './assets/css/styles.css?v=10',
  './assets/js/main.js?v=10',
  './assets/js/scan-fallback.js?v=11',
  './assets/js/scan.js?v=11',
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS_TO_PRECACHE)).catch(() => {})
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.map((k) => { if (!k.startsWith('fitlife-cache-')) return; if (k !== CACHE_NAME) return caches.delete(k); }));
      await self.clients.claim();
    })()
  );
});

function isSameOrigin(url) {
  try { return new URL(url, self.location.origin).origin === self.location.origin; } catch { return false; }
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  // Navigation requests: network-first with cache fallback (for offline)
  if (request.mode === 'navigate') {
    event.respondWith(
      (async () => {
        try {
          const fresh = await fetch(request);
          const cache = await caches.open(CACHE_NAME);
          cache.put(request, fresh.clone()).catch(() => {});
          return fresh;
        } catch (_) {
          const cache = await caches.open(CACHE_NAME);
          const cached = await cache.match('/index.html');
          return cached || new Response('<h1>Offline</h1><p>You appear to be offline.</p>', { headers: { 'Content-Type': 'text/html' } });
        }
      })()
    );
    return;
  }

  // Same-origin assets: cache-first
  if (isSameOrigin(request.url)) {
    event.respondWith(
      (async () => {
        const cache = await caches.open(CACHE_NAME);
        const cached = await cache.match(request);
        if (cached) return cached;
        try {
          const fresh = await fetch(request);
          cache.put(request, fresh.clone()).catch(() => {});
          return fresh;
        } catch (_) {
          return cached || Response.error();
        }
      })()
    );
    return;
  }
  // Cross-origin (e.g., CDN): do not intercept to avoid CORS/module issues
  return;
});

self.addEventListener('message', (event) => {
  if (event.data === 'SKIP_WAITING') self.skipWaiting();
});

