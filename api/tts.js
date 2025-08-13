import { PollyClient, SynthesizeSpeechCommand, DescribeVoicesCommand } from '@aws-sdk/client-polly';

function makeSSML(text, rate = process.env.POLLY_RATE || 'medium', pitch = process.env.POLLY_PITCH || '+4%'){
  const escape = (s)=>s.replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
  const parts = String(text||'').split(/(?<=[.!?])\s+/).map(s=>s.trim()).filter(Boolean).map(escape);
  const inner = parts.map(s=>`<s>${s}</s><break time='120ms'/>`).join("<break time='80ms'/>");
  return `<speak><prosody rate='${rate}' pitch='${pitch}'>${inner}</prosody></speak>`;
}

async function chooseVoice(client, preferred, engine){
  try{
    const out = await client.send(new DescribeVoicesCommand({}));
    const voices = out.Voices || [];
    const pref = (preferred||'').toLowerCase();
    const hasEngine = (v)=> (v.SupportedEngines||[]).includes(engine);
    // 1) exact preferred with engine
    for(const v of voices){ if((v.Id||'').toLowerCase()===pref && hasEngine(v)) return v.Id; }
    // 2) any EN Female with engine
    for(const v of voices){ if((v.LanguageCode||'').startsWith('en') && v.Gender==='Female' && hasEngine(v)) return v.Id; }
    // 3) any EN with engine
    for(const v of voices){ if((v.LanguageCode||'').startsWith('en') && hasEngine(v)) return v.Id; }
    // 4) any with engine
    for(const v of voices){ if(hasEngine(v)) return v.Id; }
  }catch{}
  return preferred || 'Ruth';
}

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  try {
    const base = process.env.BACKEND_BASE_URL;
    if (base) {
      // Prefer using your FastAPI backend if available
      const r = await fetch(`${base.replace(/\/$/, '')}/api/tts`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req.body||{})
      });
      const data = await r.json();
      return res.status(r.status).json(data);
    }

    const { text = '' } = req.body || {};
    const clean = String(text||'').trim();
    if(!clean) return res.status(400).json({ error: 'Empty text' });

    const region = process.env.POLLY_REGION || process.env.AWS_REGION || 'ap-south-1';
    const voicePref = process.env.POLLY_VOICE || 'Ruth';
    const client = new PollyClient({ region });

    // Synthesis plan (mirroring backend)
    const plan = [
      { engine:'neural', type:'ssml', text: makeSSML(clean) },
      { engine:'standard', type:'ssml', text: makeSSML(clean) },
      { engine:'neural', type:'text', text: clean },
      { engine:'standard', type:'text', text: clean },
    ];

    let audioB64=null, engineUsed=null, voiceUsed=null, lastErr=null;
    for(const step of plan){
      const voice = await chooseVoice(client, voicePref, step.engine);
      try{
        const out = await client.send(new SynthesizeSpeechCommand({
          VoiceId: voice,
          OutputFormat: 'mp3',
          Text: step.text,
          TextType: step.type.toUpperCase(),
          Engine: step.engine,
        }));
        const buf = Buffer.from(await out.AudioStream.transformToByteArray());
        audioB64 = buf.toString('base64'); engineUsed = step.engine; voiceUsed = voice; break;
      }catch(e){ lastErr = e; }
    }
    if(!audioB64){
      return res.status(500).json({ error: 'polly-synthesize-failed', detail: String(lastErr) });
    }

    // Viseme marks (best-effort)
    let marks = [];
    try{
      const out = await client.send(new SynthesizeSpeechCommand({
        VoiceId: voiceUsed,
        OutputFormat: 'json',
        Text: clean,
        TextType: 'text',
        SpeechMarkTypes: ['viseme'],
        Engine: engineUsed,
      }));
      const txt = Buffer.from(await out.AudioStream.transformToByteArray()).toString('utf8');
      marks = txt.split(/\r?\n/).filter(Boolean).map(s=>JSON.parse(s));
    }catch{}

    return res.status(200).json({ audio_b64: audioB64, marks });
  } catch (e) {
    return res.status(500).json({ error: 'tts-failed', detail: String(e) });
  }
}
