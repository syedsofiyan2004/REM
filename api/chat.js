export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  try {
    const { text = '', session_id = 'vercel' } = req.body || {};
    // Simple local reply for demo (no AWS). Adjust if you want to call a backend.
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
