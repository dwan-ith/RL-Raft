let latest = null;

async function fetchSnapshot() {
  try {
    const res = await fetch('/api/snapshot');
    latest = await res.json();
    render(latest);
  } catch (e) {
    console.error(e);
  }
}

async function sendCommand(payload) {
  await fetch('/api/control', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload)
  });
  setTimeout(fetchSnapshot, 150);
}

function nodeAction(action) {
  sendCommand({ action, node_id: Number(document.getElementById('nodeSelect').value) });
}

function updateNetwork() {
  sendCommand({
    action: 'network',
    base_latency_ms: Number(document.getElementById('latency').value),
    latency_jitter_ms: Number(document.getElementById('jitter').value),
    packet_loss: Number(document.getElementById('loss').value)
  });
}

function majorityPartition() {
  if (!latest) return;
  const ids = Object.keys(latest.nodes).map(Number).sort((a, b) => a - b);
  const mid = Math.ceil(ids.length / 2) + 1;
  sendCommand({ action: 'partition', groups: [ids.slice(0, mid), ids.slice(mid)] });
}

function render(s) {
  const m = s.metrics;
  document.getElementById('subtitle').textContent = `policy=${s.config.policy_mode} leader=${m.leader_id ?? 'none'} term=${m.current_term}`;

  const select = document.getElementById('nodeSelect');
  const selected = select.value;
  select.innerHTML = Object.keys(s.nodes).map(id => `<option value="${id}">Node ${id}</option>`).join('');
  if (selected && Object.keys(s.nodes).includes(selected)) select.value = selected;

  document.getElementById('nodes').innerHTML = Object.values(s.nodes).map(n => `
    <div class="node ${n.state === 'leader' ? 'leader' : ''} ${n.active ? '' : 'dead'}">
      <div class="node-title"><span>Node ${n.node_id}</span><span class="pill">${n.active ? 'active' : 'crashed'}</span></div>
      <div class="role">${n.state}</div>
      <div>term: ${n.term}</div>
      <div>ldr: ${n.leader_id ?? '-'}</div>
      <div>log: ${n.log_len}</div>
      <div>rtt: ${Math.round(n.estimated_rtt_ms || 0)}ms</div>
    </div>`).join('');

  const metricRows = [
    ['Nodes', Object.keys(s.nodes).length],
    ['Leader', m.leader_id ?? 'none'],
    ['Elections', m.elections_started],
    ['Split votes', m.split_votes],
    ['Delivered', m.messages_delivered],
    ['Dropped', m.messages_dropped],
    ['Failover s', fmt(m.last_failover_time_s)],
    ['Stability s', fmt(m.leader_stability_s)]
  ];
  document.getElementById('metrics').innerHTML = metricRows.map(([k, v]) => `<div class="metric">${k}<b>${v}</b></div>`).join('');
  document.getElementById('events').innerHTML = s.recent_events.slice().reverse().map(e => `<div>${e.wall_time || ''} ${e.event} ${JSON.stringify(e)}</div>`).join('');

  const net = s.network;
  if (document.activeElement !== document.getElementById('latency')) document.getElementById('latency').value = net.base_latency_ms ?? 0;
  if (document.activeElement !== document.getElementById('jitter')) document.getElementById('jitter').value = net.latency_jitter_ms ?? 0;
  if (document.activeElement !== document.getElementById('loss')) document.getElementById('loss').value = net.packet_loss ?? 0;
}

function fmt(x) { return x == null ? '-' : Number(x).toFixed(3); }

setInterval(fetchSnapshot, 500);
fetchSnapshot();
