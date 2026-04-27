// KV cache observer client.
//
// 1. GET /observer/snapshot to bootstrap the grid (one cell per physical block).
// 2. Open WS /observer/ws and apply alloc/free/hit/append events as they arrive.
// 3. Hovering a cell pre-renders the detail pane; clicking pins it.

const grid = document.getElementById('grid');
const detailEmpty = document.getElementById('detail-empty');
const detailFields = document.getElementById('detail-fields');
const dId = document.getElementById('d-id');
const dRef = document.getElementById('d-refcount');
const dHash = document.getElementById('d-hash');
const dOwners = document.getElementById('d-owners');
const dTokens = document.getElementById('d-tokens');
const dText = document.getElementById('d-text');

const blocks = [];   // block_id -> {refcount, hash, owners[], tokens[]}
const cells = [];    // block_id -> DOM element

let pinnedBlockId = null;

function classFor(b) {
  if (!b) return 'block';
  if (b.refcount > 1) return 'block shared';
  if (b.refcount === 1) return 'block owned';
  // refcount=0: distinguish "free fresh" from "evictable but holds cached content".
  if (b.tokens && b.tokens.length > 0) return 'block evictable';
  return 'block';
}

function renderDetail(blockId, decodedText) {
  const b = blocks[blockId];
  if (!b) {
    detailEmpty.hidden = false;
    detailFields.hidden = true;
    return;
  }
  detailEmpty.hidden = true;
  detailFields.hidden = false;
  dId.textContent = blockId;
  dRef.textContent = b.refcount;
  dHash.textContent = b.block_hash != null ? '0x' + b.block_hash.toString(16) : '(partial)';
  dOwners.textContent = (b.owners || []).join(', ') || '—';
  dTokens.textContent = (b.tokens || []).map(t => `[${t}]`).join(' ') || '—';
  if (decodedText !== undefined) {
    dText.textContent = decodedText || '(empty)';
  } else if (!b.tokens || b.tokens.length === 0) {
    dText.textContent = '—';
  } else {
    dText.textContent = 'click to decode';
  }
}

async function decodeAndShow(blockId) {
  const b = blocks[blockId];
  if (!b || !b.tokens || b.tokens.length === 0) return;
  dText.textContent = 'decoding…';
  try {
    const resp = await fetch(`/observer/decode?ids=${b.tokens.join(',')}`);
    if (!resp.ok) throw new Error(`status ${resp.status}`);
    const json = await resp.json();
    if (pinnedBlockId === blockId) {
      // Only render if user hasn't moved on.
      renderDetail(blockId, json.text);
    }
  } catch (err) {
    dText.textContent = `decode failed: ${err}`;
  }
}

function flash(cell) {
  cell.classList.add('flash');
  setTimeout(() => cell.classList.remove('flash'), 450);
}

function updateCounts() {
  let free = 0, evictable = 0, shared = 0;
  for (const b of blocks) {
    if (!b || b.refcount === 0) {
      if (b && b.tokens && b.tokens.length > 0) evictable++;
      else free++;
    } else if (b.refcount > 1) shared++;
  }
  document.getElementById('free-count').textContent = free;
  document.getElementById('shared-count').textContent = shared;
  const ev = document.getElementById('evictable-count');
  if (ev) ev.textContent = evictable;
}

function applyEvent(ev) {
  // ev = {kind, block_id, seq_id, refcount, block_hash, tokens, layer_group}
  let b = blocks[ev.block_id];
  if (!b) {
    b = blocks[ev.block_id] = { refcount: 0, block_hash: null, owners: [], tokens: [] };
  }
  b.refcount = ev.refcount;
  b.block_hash = ev.block_hash;

  // 'evict' clears the slot for the new owner; before applying their alloc
  // the contents are gone. We rely on the alloc event that follows.
  if (ev.kind === 'evict') {
    b.tokens = [];
    b.owners = [];
  } else {
    b.tokens = ev.tokens || [];
    const owners = new Set(b.owners);
    if (ev.kind === 'alloc' || ev.kind === 'hit') {
      owners.add(ev.seq_id);
    } else if (ev.kind === 'release') {
      owners.delete(ev.seq_id);
    }
    b.owners = [...owners];
  }

  const cell = cells[ev.block_id];
  if (cell) {
    cell.className = classFor(b);
    // Red flash on evict (slot just got repurposed — the previous occupant lost it).
    if (ev.kind === 'evict') flash(cell);
  }
  updateCounts();
  if (pinnedBlockId === ev.block_id) renderDetail(ev.block_id);
}

async function bootstrap() {
  const resp = await fetch('/observer/snapshot');
  const snap = await resp.json();
  document.getElementById('model-id').textContent = snap.model;
  document.getElementById('block-size').textContent = snap.block_size;
  document.getElementById('block-count').textContent = snap.num_blocks;
  for (const b of snap.blocks) {
    blocks[b.block_id] = b;
  }
  // Build cells.
  grid.innerHTML = '';
  for (let i = 0; i < snap.num_blocks; i++) {
    const cell = document.createElement('div');
    cell.dataset.blockId = i;
    cell.title = `block ${i}`;
    cell.className = classFor(blocks[i]);
    cell.addEventListener('mouseenter', () => {
      if (pinnedBlockId === null) renderDetail(i);
    });
    cell.addEventListener('click', () => {
      if (pinnedBlockId === i) {
        pinnedBlockId = null;
        renderDetail(null);
      } else {
        pinnedBlockId = i;
        renderDetail(i);
        decodeAndShow(i);
      }
    });
    cells[i] = cell;
    grid.appendChild(cell);
  }
  updateCounts();
}

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${location.host}/observer/ws`);
  ws.addEventListener('message', e => {
    try {
      applyEvent(JSON.parse(e.data));
    } catch (err) {
      console.error('bad event', err, e.data);
    }
  });
  ws.addEventListener('close', () => {
    setTimeout(connectWS, 500);  // auto-reconnect.
  });
}

bootstrap().then(connectWS);

// ---- one-click demo buttons ------------------------------------------------

// Demo-button payloads.
//
// The prompt files live on disk under /prompts/{N}.txt and are fetched
// lazily on click. Each one is a deterministic line-jumping graph (lines
// like "331: 5124" — line number colon target) plus a short instruction
// to start at line 1 and follow N jumps. Forces the model to re-quote
// rules from context, so KV cache fills with the rules block and you can
// watch it grow.

// Qwen3 emits a <think>…</think> reasoning block before the answer, so we
// need enough budget for thinking PLUS the path. Each "i: target" step is
// ~5-7 tokens. Path length is N/2 so jumps = N/2 - 1.
const TOKENS_BY_SIZE = {
  '10': 400,        // 4 jumps + thinking
  '100': 1500,      // 49 jumps + thinking
  '1000': 6000,     // 499 jumps + thinking; close to 40k context with prompt
  '10000': 10000,   // exceeds context — will hit limit; demo of scale
  shared: 800,
};

async function loadPrompt(name) {
  // `name` is one of '10' / '100' / '1000' / '10000', or 'shared' which loads 100.txt.
  const file = name === 'shared' ? '100' : name;
  const resp = await fetch(`/prompts/${file}.txt`);
  if (!resp.ok) throw new Error(`prompt ${file}.txt not found (status ${resp.status})`);
  return resp.text();
}

const status = document.getElementById('chat-status');
const output = document.getElementById('chat-output');
const buttons = document.querySelectorAll('button.demo');

// Each demo run gets its own <div> in the output column and streams tokens
// in via SSE. Multiple runs can be in flight simultaneously — clicking
// another button does not interrupt or block existing ones.

let runCounter = 0;
let inflight = 0;

function newRunCard(label) {
  const card = document.createElement('div');
  card.className = 'run';
  const head = document.createElement('div');
  head.className = 'run-head';
  head.textContent = label;
  const body = document.createElement('div');
  body.className = 'run-body';
  card.appendChild(head);
  card.appendChild(body);
  output.appendChild(card);
  output.scrollTop = output.scrollHeight;
  return { head, body };
}

function setRunning(state) {
  status.textContent = state ? `${inflight} request${inflight === 1 ? '' : 's'} in flight` : 'idle';
}

async function streamChat(prompt, max_tokens, label) {
  inflight++;
  setRunning(true);
  const id = ++runCounter;
  const { head, body } = newRunCard(`#${id} ${label} starting…`);
  const t0 = performance.now();
  let firstTokenAt = null;
  let n_chars = 0;
  let n_chunks = 0;
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: document.getElementById('model-id').textContent,
        messages: [{ role: 'user', content: prompt }],
        max_tokens,
        temperature: 0.0,
        stream: true,
      }),
    });
    if (!res.body) throw new Error('no response body');
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === '[DONE]') continue;
        try {
          const obj = JSON.parse(payload);
          const delta = obj.choices?.[0]?.delta?.content;
          if (delta) {
            if (firstTokenAt === null) firstTokenAt = performance.now();
            body.textContent += delta;
            n_chars += delta.length;
            n_chunks++;
          }
        } catch (_) { /* ignore malformed chunk */ }
      }
      output.scrollTop = output.scrollHeight;
    }
    const dt = (performance.now() - t0) / 1000;
    const ttft = firstTokenAt !== null ? ((firstTokenAt - t0) / 1000).toFixed(2) : '?';
    const decode_dt = firstTokenAt !== null ? (performance.now() - firstTokenAt) / 1000 : 0;
    const tps = decode_dt > 0 ? (n_chunks / decode_dt).toFixed(1) : '?';
    head.textContent = `#${id} ${label} · TTFT ${ttft}s · ${dt.toFixed(2)}s · ${n_chunks} tokens · ${tps} tok/s`;
  } catch (err) {
    head.textContent = `#${id} ${label} · error: ${err}`;
  } finally {
    inflight--;
    setRunning(inflight > 0);
  }
}

async function runDemo(name) {
  let text;
  try {
    text = await loadPrompt(name === 'shared' ? 'shared' : name);
  } catch (err) {
    newRunCard(`error loading prompt: ${err}`);
    return;
  }
  if (name === 'shared') {
    streamChat(text, TOKENS_BY_SIZE.shared, 'shared req A');
    streamChat(text, TOKENS_BY_SIZE.shared, 'shared req B');
  } else {
    streamChat(text, TOKENS_BY_SIZE[name] || 200, `${name} lines`);
  }
}

for (const btn of buttons) {
  btn.addEventListener('click', () => runDemo(btn.dataset.demo));
}
