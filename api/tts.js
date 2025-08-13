export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  try {
    // No server TTS on Vercel free demo; instruct client to use browser speech
    return res.status(501).json({ error: 'use-browser-tts' });
  } catch (e) {
    return res.status(500).json({ error: 'tts-failed' });
  }
}
