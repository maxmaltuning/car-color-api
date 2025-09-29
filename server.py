// server.js
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fetch = (...args) => import('node-fetch').then(({default: f}) => f(...args));
require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json({limit: "20mb"}));

const upload = multer({ storage: multer.memoryStorage() });

const REPLICATE_TOKEN = process.env.REPLICATE_API_TOKEN;
const MODEL_VERSION = "7b2c0e9f0dfeeddc8b20ffec90dd1df6acf9c8d3b0848bb0d9a7d5f2aa4e2d8c"; // lucataco/segment-anything-2

if (!REPLICATE_TOKEN) {
  console.error("⚠️  No REPLICATE_API_TOKEN in .env");
}

async function uploadToReplicate(fileBuf, mime = "image/png") {
  const fd = new (require("form-data"))();
  fd.append("file", fileBuf, { filename: "image.png", contentType: mime });
  const r = await fetch("https://api.replicate.com/v1/files", {
    method: "POST",
    headers: { Authorization: `Token ${REPLICATE_TOKEN}` },
    body: fd
  });
  const j = await r.json();
  if (!r.ok) throw new Error("Upload failed: " + JSON.stringify(j));
  return j.url; // https://replicate.delivery/pbxt/...
}

async function startPrediction(imageUrl) {
  const body = {
    version: MODEL_VERSION,
    input: { image: imageUrl }
  };
  const r = await fetch("https://api.replicate.com/v1/predictions", {
    method: "POST",
    headers: {
      Authorization: `Token ${REPLICATE_TOKEN}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
  const j = await r.json();
  if (!r.ok) throw new Error("Prediction start failed: " + JSON.stringify(j));
  return j.urls.get; // polling URL
}

async function pollUntilDone(getUrl, timeoutMs = 120000, stepMs = 2000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const r = await fetch(getUrl, { headers: { Authorization: `Token ${REPLICATE_TOKEN}` }});
    const j = await r.json();
    if (j.status === "succeeded") return j;
    if (j.status === "failed" || j.status === "canceled") throw new Error("Prediction " + j.status);
    await new Promise(s => setTimeout(s, stepMs));
  }
  throw new Error("Prediction timeout");
}

app.get("/", (_,res)=>res.send("Server is running"));

/**
 * POST /segment
 * приймає або file (multipart), або {image:"https://..."} у JSON
 * повертає {status, output} де output — URL(и) масок
 */
app.post("/segment", upload.single("file"), async (req, res) => {
  try {
    let imageUrl = null;

    if (req.file) {
      imageUrl = await uploadToReplicate(req.file.buffer, req.file.mimetype);
    } else if (req.is("application/json") && req.body?.image) {
      imageUrl = req.body.image;
    } else {
      return res.status(400).json({ error: "Provide image file or {image: 'https://...'}" });
    }

    const getUrl = await startPrediction(imageUrl);
    const result = await pollUntilDone(getUrl);

    // На виході у SAM2 зазвичай масив URLів (PNG маски)
    return res.json({
      status: result.status,
      output: result.output || null
    });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, ()=> console.log(`✅ Server running on http://localhost:${PORT}`));
