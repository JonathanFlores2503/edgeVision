// Service worker: cachea el "app shell" para que la app cargue al instante y se
// pueda instalar en el teléfono. Los datos en vivo (stream/SSE/API) van siempre a
// la red local (la Jetson); aquí solo se cachea la interfaz estática.
const CACHE = "edgevision-v1";
const SHELL = ["/", "/manifest.webmanifest", "/icon-192.png", "/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  // Nunca cachear datos en vivo: stream de vídeo, eventos SSE y API.
  if (url.pathname.startsWith("/stream/") ||
      url.pathname === "/events" ||
      url.pathname.startsWith("/api/")) {
    return; // va directo a la red
  }
  // HTML (navegación): network-first para tomar siempre la última interfaz,
  // con respaldo a caché si no hay red (offline en el vehículo).
  if (e.request.mode === "navigate" || url.pathname === "/") {
    e.respondWith(
      fetch(e.request)
        .then((resp) => {
          caches.open(CACHE).then((c) => c.put("/", resp.clone()));
          return resp;
        })
        .catch(() => caches.match("/"))
    );
    return;
  }
  // Resto del app shell (iconos/manifest): cache-first con respaldo a red.
  e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request)));
});
