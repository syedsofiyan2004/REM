export const config = { runtime: 'edge' };

export default async function handler(req) {
  if (req.method !== 'POST') return new Response('Method not allowed', { status: 405 });
  const { text = '', session_id = 'vercel' } = await req.json();

  // If a backend is configured, proxy the request to it and pass through the stream
  const base = process.env.BACKEND_BASE_URL;
  if (base) {
    const r = await fetch(`${base.replace(/\/$/, '')}/api/chat_stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, session_id })
    });
    // Pass-through response as a stream
    return new Response(r.body, { headers: { 'Content-Type': 'application/jsonl; charset=utf-8' } });
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      // Simulate a short streaming response
      const reply = simulate(text);
      for (const ch of reply) {
        controller.enqueue(encoder.encode(JSON.stringify({ delta: ch }) + '\n'));
        await new Promise(r => setTimeout(r, 15));
      }
      controller.close();
    }
  });
  return new Response(stream, { headers: { 'Content-Type': 'application/jsonl; charset=utf-8' }});
}

function simulate(input) {
  const t = String(input || '').trim();
  if (!t) return "I'm here.";
  if (/name/i.test(t)) return 'Rem.';
  if (/(hi|hello|hey)\b/i.test(t)) return 'Hi! How can I help?';
  return t.length < 60 ? `You said: ${t}` : 'Got it. What\'s next?';
}
