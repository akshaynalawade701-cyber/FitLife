self.addEventListener('install',e=>self.skipWaiting());
self.addEventListener('activate',e=>{e.waitUntil(self.registration.unregister().then(()=>clients.matchAll({type:'window'})).then(ws=>Promise.all(ws.map(w=>w.navigate(w.url)))));});
self.addEventListener('fetch',()=>{});