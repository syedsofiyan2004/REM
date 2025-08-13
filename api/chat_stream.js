export const config = { runtime: 'edge' };

export default async function handler(req) {
  if (req.method !== 'POST') return new Response('Method not allowed', { status: 405 });
  const { text = '' } = await req.json();
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
