export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  try {
    const base = process.env.BACKEND_BASE_URL;
    if (base) {
      // Proxy to your FastAPI backend
      const r = await fetch(`${base.replace(/\/$/, '')}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req.body || {})
      });
      const data = await r.json();
      return res.status(r.status).json(data);
    }

    const { text = '', session_id = 'vercel' } = req.body || {};
    // Local fallback (no AWS)
    const lower = text.trim().toLowerCase();
    let reply = 'I\'m here.';
    if (!text.trim()) reply = "I\'m here.";
    else if (/name/.test(lower)) reply = 'Rem.';
    else if (/(hi|hello|hey)\b/.test(lower)) reply = 'Hi! How can I help?';
    else reply = text.length < 60 ? `You said: ${text}` : 'Got it. What\'s next?';
    res.status(200).json({ reply });
  } catch (e) {
    res.status(500).json({ error: 'chat-failed' });
  }
}
