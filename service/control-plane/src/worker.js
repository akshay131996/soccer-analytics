/**
 * Control plane between the Android app and the RunPod endpoint.
 *
 * It exists for one reason: the app cannot hold secrets. The RunPod API key and the
 * R2 bucket credentials live here, on the server, and the app gets short-lived
 * capability URLs instead.
 *
 * Endpoints
 *   POST   /jobs                 -> { job_id, upload_url }
 *   POST   /jobs/:id/start       -> { status } (dispatches to RunPod)
 *   GET    /jobs/:id             -> { status, progress, stats?, result_url?, error? }
 *   PUT    /blob/:key?token=...  -> streams a body into R2   (app upload, worker output)
 *   GET    /blob/:key?token=...  -> streams an object out of R2 (worker input, app download)
 *
 * The /blob routes are guarded by an HMAC token rather than S3 presigning, which keeps
 * the whole thing to one dependency-free file. The token binds the key, the method and
 * an expiry, so a URL minted for reading cannot be replayed to write.
 *
 * RunPod Serverless already provides async jobs -- /run returns an id, /status reports
 * progress -- so this deliberately stores almost nothing. KV holds one small record
 * per job purely so the app never sees a RunPod id.
 */

const JOB_TTL_SECONDS = 60 * 60 * 24;      // matches the R2 lifecycle rule
const UPLOAD_WINDOW = 60 * 30;             // 30 min to finish an upload
const WORKER_WINDOW = 60 * 60;             // the worker's own URLs

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json" },
  });

/* ---------- capability tokens ---------- */

async function sign(env, key, method, exp) {
  const data = new TextEncoder().encode(`${method}:${key}:${exp}`);
  const k = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(env.SIGNING_SECRET),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", k, data);
  return [...new Uint8Array(sig)].map(b => b.toString(16).padStart(2, "0")).join("");
}

async function mint(env, base, key, method, window) {
  const exp = Math.floor(Date.now() / 1000) + window;
  const token = await sign(env, key, method, exp);
  return `${base}/blob/${encodeURIComponent(key)}?exp=${exp}&token=${token}&m=${method}`;
}

async function verify(env, url, key, method) {
  const exp = Number(url.searchParams.get("exp") || 0);
  const token = url.searchParams.get("token") || "";
  if (url.searchParams.get("m") !== method) return false;
  if (!exp || exp < Math.floor(Date.now() / 1000)) return false;
  const expected = await sign(env, key, method, exp);
  // Constant-time-ish: compare full strings of equal length only.
  if (token.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < token.length; i++) diff |= token.charCodeAt(i) ^ expected.charCodeAt(i);
  return diff === 0;
}

/* ---------- handlers ---------- */

async function createJob(env, base) {
  const id = crypto.randomUUID();
  const key = `in/${id}.mp4`;
  await env.JOBS.put(id, JSON.stringify({ state: "awaiting_upload" }),
                     { expirationTtl: JOB_TTL_SECONDS });
  return json({
    job_id: id,
    upload_url: await mint(env, base, key, "PUT", UPLOAD_WINDOW),
  });
}

async function startJob(env, base, id, body) {
  const raw = await env.JOBS.get(id);
  if (!raw) return json({ error: "unknown job" }, 404);

  const inKey = `in/${id}.mp4`;
  const outKey = `out/${id}.mp4`;

  const head = await env.BUCKET.head(inKey);
  if (!head) return json({ error: "upload not found — PUT the video first" }, 409);

  const payload = {
    input: {
      video_url: await mint(env, base, inKey, "GET", WORKER_WINDOW),
      upload_url: await mint(env, base, outKey, "PUT", WORKER_WINDOW),
      keypoints: body.keypoints ?? null,
      imgsz: body.imgsz ?? 1280,
      conf: body.conf ?? 0.3,
    },
  };

  const res = await fetch(`https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/run`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${env.RUNPOD_API_KEY}`,
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    return json({ error: "dispatch failed", detail: await res.text() }, 502);
  }
  const { id: runpodId } = await res.json();
  await env.JOBS.put(id, JSON.stringify({ state: "queued", runpodId }),
                     { expirationTtl: JOB_TTL_SECONDS });
  return json({ status: "queued" });
}

async function getJob(env, base, id) {
  const raw = await env.JOBS.get(id);
  if (!raw) return json({ error: "unknown job" }, 404);
  const rec = JSON.parse(raw);
  if (!rec.runpodId) return json({ status: rec.state });

  const res = await fetch(
    `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/status/${rec.runpodId}`,
    { headers: { authorization: `Bearer ${env.RUNPOD_API_KEY}` } }
  );
  if (!res.ok) return json({ error: "status check failed" }, 502);
  const rp = await res.json();

  // RunPod: IN_QUEUE | IN_PROGRESS | COMPLETED | FAILED | CANCELLED | TIMED_OUT
  const out = { status: rp.status, progress: rp.output?.progress ?? rp.stream ?? null };

  if (rp.status === "COMPLETED") {
    const result = rp.output || {};
    if (result.ok === false) {
      // A handled pipeline failure. Surface the code so the app can say something
      // specific rather than "processing failed".
      return json({ status: "FAILED", code: result.code, error: result.message });
    }
    out.stats = result.stats ?? null;
    out.result_url = await mint(env, base, `out/${id}.mp4`, "GET", WORKER_WINDOW);
  } else if (rp.status === "FAILED" || rp.status === "TIMED_OUT") {
    out.error = typeof rp.error === "string" ? rp.error : "the worker failed";
  }
  return json(out);
}

async function blob(env, request, url, key) {
  if (request.method === "PUT") {
    if (!(await verify(env, url, key, "PUT"))) return json({ error: "bad token" }, 403);
    await env.BUCKET.put(key, request.body);          // streamed, never buffered
    return json({ ok: true });
  }
  if (request.method === "GET") {
    if (!(await verify(env, url, key, "GET"))) return json({ error: "bad token" }, 403);
    const obj = await env.BUCKET.get(key);
    if (!obj) return json({ error: "not found" }, 404);
    return new Response(obj.body, {
      headers: { "content-type": "video/mp4", "content-length": String(obj.size) },
    });
  }
  return json({ error: "method not allowed" }, 405);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const base = `${url.protocol}//${url.host}`;
    const parts = url.pathname.split("/").filter(Boolean);

    try {
      if (parts[0] === "blob" && parts[1]) {
        return await blob(env, request, url, decodeURIComponent(parts[1]));
      }
      if (parts[0] === "jobs") {
        if (request.method === "POST" && parts.length === 1) {
          return await createJob(env, base);
        }
        if (request.method === "POST" && parts[2] === "start") {
          return await startJob(env, base, parts[1], await request.json().catch(() => ({})));
        }
        if (request.method === "GET" && parts.length === 2) {
          return await getJob(env, base, parts[1]);
        }
      }
      return json({ error: "not found" }, 404);
    } catch (e) {
      return json({ error: "internal", detail: String(e) }, 500);
    }
  },
};
